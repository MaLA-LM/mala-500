"""
CoQA: A Conversational Question Answering Challenge
https://arxiv.org/pdf/1808.07042.pdf

CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

Homepage: https://stanfordnlp.github.io/coqa/
"""
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.api.metric import mean
from lm_eval.api.task import PromptSourceTask

_CITATION = """
@misc{reddy2018coqa,
    title={CoQA: A Conversational Question Answering Challenge},
    author={Siva Reddy and Danqi Chen and Christopher D. Manning},
    year={2018},
    eprint={1808.07042},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class Sib200_test(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "sib200"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["test"]
        # return self.dataset["validation"]

