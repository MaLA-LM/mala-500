import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import pathlib 

import datasets
from datasets import load_dataset

import transformers
from transformers import EarlyStoppingCallback
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    LlamaTokenizer
)

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def load_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    tokenizer_class = LlamaTokenizer if "llama" in model_args.model_name_or_path else AutoTokenizer
    
    if model_args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        print(f"✅ load tokenizer from: {model_args.tokenizer_name}")
    elif model_args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        print(f"✅ load tokenizer from: {model_args.model_name_or_path}")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    

    return tokenizer


def load_data(data_args, model_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
    
    print("✅ Loaded Raw Dataset:")
    print(raw_datasets)
    return raw_datasets


def preprocess_data(training_args, data_args, model_args, tokenizer):
    with training_args.main_process_first(desc="dataset map tokenization"):
        # cache tokenized data
        if data_args.dataset_name is not None:
            base_cache_dir = f"{model_args.cache_dir}/{data_args.dataset_name}/{data_args.dataset_config_name}"
            saved_tokenized_datasets_fp = pathlib.Path(f"{base_cache_dir}/{os.path.basename(model_args.model_name_or_path)}_tokenized_data_{data_args.max_train_samples}train_{data_args.max_eval_samples}_eval_{len(tokenizer)}vocab.pt")
        else:
            base_cache_dir = pathlib.Path(f"{data_args.train_file}")
            saved_tokenized_datasets_fp = base_cache_dir.parent / f"{os.path.basename(model_args.model_name_or_path)}_tokenized_data_{base_cache_dir.name}_{len(tokenizer)}vocab.pt"

        if not data_args.overwrite_cache and saved_tokenized_datasets_fp.exists() and saved_tokenized_datasets_fp.is_file():
            tokenized_datasets = torch.load(str(saved_tokenized_datasets_fp))
            logger.info(f"✅ loaded tokenized_data from {saved_tokenized_datasets_fp}")
        else:
            raw_datasets = load_data(data_args, model_args)
                                                      
            # First we tokenize all the texts.
            if training_args.do_train:
                column_names = raw_datasets["train"].column_names
            else:
                column_names = raw_datasets["validation"].column_names

            text_column_name = data_args.data_column_names if data_args.data_column_names in column_names else column_names[0]
            # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
            tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

            def tokenize_function(examples):
                with CaptureLogger(tok_logger) as cl:
                    output = tokenizer(examples[text_column_name])
                    # clm input could be much much longer than block_size
                    if "Token indices sequence length is longer than the" in cl.out:
                        tok_logger.warning(
                            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                        )
                    return output
            
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
                                                      
            torch.save(tokenized_datasets, saved_tokenized_datasets_fp)
            logger.info(f"✅ saved tokenized_data to {saved_tokenized_datasets_fp}")

        if "train" not in tokenized_datasets and training_args.do_train:
            raise ValueError("--do_train requires a train dataset")
        if "validation" not in tokenized_datasets and training_args.do_eval:
            raise ValueError("--do_eval requires a validation dataset")

        if data_args.count_training_tokens:
            from tqdm import tqdm
            count_total_tokens = 0
            for i in tqdm(range(len(tokenized_datasets['train'])), desc="counting training tokens"):
                count_total_tokens += len(tokenized_datasets['train'][i]['input_ids'])
            print("Total training tokens:", count_total_tokens)
            assert False
        return tokenized_datasets


def get_lm_dataset(training_args, data_args, model_args, tokenizer):
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if data_args.dataset_name is not None:
            base_cache_dir = f"{model_args.cache_dir}/{data_args.dataset_name}/{data_args.dataset_config_name}"
            if data_args.block_size == 2048:
                saved_lm_datasets_fp = pathlib.Path(f"{base_cache_dir}/{os.path.basename(model_args.model_name_or_path)}_lm_data_{data_args.max_train_samples}train_{data_args.max_eval_samples}eval_{len(tokenizer)}vocab.pt")
            else:
                saved_lm_datasets_fp = pathlib.Path(f"{base_cache_dir}/{os.path.basename(model_args.model_name_or_path)}_lm_data_{data_args.max_train_samples}train_{data_args.max_eval_samples}eval_{len(tokenizer)}vocab_{data_args.block_size}.pt")
        else:
            base_cache_dir = pathlib.Path(f"{data_args.train_file}")
            if data_args.block_size == 2048:
                saved_lm_datasets_fp = base_cache_dir.parent / f"{os.path.basename(model_args.model_name_or_path)}_lm_data_{base_cache_dir.name}_{len(tokenizer)}vocab.pt"        
            else:
                saved_lm_datasets_fp = base_cache_dir.parent / f"{os.path.basename(model_args.model_name_or_path)}_lm_data_{base_cache_dir.name}_{len(tokenizer)}vocab_{data_args.block_size}.pt"        

        if not data_args.overwrite_cache and saved_lm_datasets_fp.exists() and saved_lm_datasets_fp.is_file():
            lm_datasets = torch.load(str(saved_lm_datasets_fp))
            logger.info(f"✅ loaded lm_data from {saved_lm_datasets_fp}")
        else:
            tokenized_datasets = preprocess_data(training_args, data_args, model_args, tokenizer)
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
            torch.save(lm_datasets, saved_lm_datasets_fp)
            logger.info(f"✅ saved lm_data to {saved_lm_datasets_fp}")
        
        if data_args.count_training_tokens:
            tokenized_datasets = preprocess_data(training_args, data_args, model_args, tokenizer)
    return lm_datasets


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default="",
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    tie_word_embeddings: int = field(
        default=1,
        metadata={"help": "1: tie. 0: decoupled embedding"},
    )
    lang_adapt_strategies: str = field(
        default=None,
        metadata={"help": "language adaptation strategies"},
    )

    reinit: int = field(
        default=0,
        metadata={"help": "reinitialize the model"},
    )

    embedding_strategies: str = field(
        default="",
        metadata={"help": "choose one of the two strategies - 'replace', 'extend', 'extend-init', 'overlap-replace'"},
    )
    
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_lang: Optional[str] = field(
        default=None, metadata={"help": "The language of the dataset"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_column_names: Optional[str] = field(
        default="text", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    count_training_tokens: bool = field(
        default=False, metadata={"help": "Count the number of tokens in the training samples"}
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class PeftTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None, metadata={"help": "path to pretrained peft model"})



def load_model(model_args, tokenizer):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
        "tie_word_embeddings": bool(model_args.tie_word_embeddings), # Note Llama 2 use untied word embeddings
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        print(f"config: ")
        print(config)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token,
        )
        model.config.use_cache = False # fix the incompatibility with --gradient_checkpointing
        print(f"✅ load model from: {model_args.model_name_or_path}")
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        print(f"✅ load model from config: ")
        print(config)

    return model

# Copied from https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.data_dir = f'{training_args.output_dir}'

    # conditional checks
    assert model_args.lang_adapt_strategies in ("continual-pretrain", "lora")
    assert model_args.embedding_strategies in ("extend", "original")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}; "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"model_args {model_args}")
    logger.info(f"data_args {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            pass
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = load_tokenizer(model_args)
    model = load_model(model_args, tokenizer)

    if model_args.embedding_strategies == 'extend':
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        print(f"Original vocabulary size: {model_vocab_size}")
        print(f"len(tokenizer): {len(tokenizer)}")
        if len(tokenizer) > model_vocab_size:
            print(f"New vocabulary size, i.e., len(tokenizer): {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
            n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
            print(f"Number of parameters {n_params}")
        model_output_size = model.get_output_embeddings().weight.size(0)
        print(f"Output embeddings size: {model_output_size}")

    if model_args.lang_adapt_strategies == 'lora':
        print("Init new peft model")
        target_modules = training_args.trainable.split(',')
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        print(f"target_modules: {target_modules}")
        print(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing the datasets.
    lm_datasets = get_lm_dataset(training_args, data_args, model_args, tokenizer)
    if training_args.do_train:
        train_dataset = lm_datasets["train"]

    if training_args.do_eval:
        eval_dataset = lm_datasets["validation"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        **({}),
        **({})
    )
    if model_args.lang_adapt_strategies == 'lora':
        trainer.add_callback(SavePeftModelCallback)

    print("Model: 👇")
    print(model)
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


if __name__ == "__main__":
    main()
