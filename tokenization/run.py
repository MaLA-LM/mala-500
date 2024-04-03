import argparse, sys, logging, copy, os, json, codecs
from datetime import datetime
import sentencepiece as spm
import sentencepiece_model_pb2 as sp_model
from os import listdir
from transformers import LlamaTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--vocab_size', type=int, default=32000)

    args = parser.parse_args()
    
    # Train a new tokenizer
    input_fname = args.input_fname
    output_fname = args.save_directory + 'LLM500'
    cmd = '--input={} --model_prefix={} --vocab_size={} --model_type=bpe --character_coverage=1.0 --train_extremely_large_corpus=true'.format(input_fname, output_fname, args.vocab_size)
    spm.SentencePieceTrainer.train(cmd)

    # Load new spm
    new_m = sp_model.ModelProto()
    new_m.ParseFromString(open(output_fname + '.model', 'rb').read())
    print('Vocab size of new spm', len(new_m.pieces))
    
    # Load pretrained LLaMA 2 spm
    old_m = sp_model.ModelProto()
    old_m.ParseFromString(open(args.save_directory + 'tokenizer.model', 'rb').read())
    print('Vocab size of old spm', len(old_m.pieces))

    # Combine new/old spms
    add_cnt = 0 
    piece_d = {piece.piece: 0 for piece in old_m.pieces}
    for new_piece in new_m.pieces:
        if new_piece.piece not in piece_d:
            piece_to_add = sp_model.ModelProto().SentencePiece()
            # Add token
            piece_to_add.piece = new_piece.piece
            # Add token log-prob
            piece_to_add.score = new_piece.score
            old_m.pieces.append(piece_to_add)
            add_cnt += 1
    print('Vocab size of final spm: %s, with %s new tokens' % (len(old_m.pieces), add_cnt))
   
    # Save final spm
    final_spm_save_dir = args.save_directory + 'LLM500_tokenizer.model'
    with open(final_spm_save_dir, 'wb') as f:
        f.write(old_m.SerializeToString())
   
    # Load old tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    print(tokenizer)
    print(tokenizer.tokenize("Workers must often get their superiors' approval for any decisions they make, and are expected to obey their superiors' instructions without question."))
    print(tokenizer.tokenize("ངལ་རྩོལ་པར་ངེས་པར་དུ་རང་ཉིད་ཀྱིས་གཏན་འབེབས་བྱས་པའི་ཆོད་དོན་གང་ཞིག་ཡིན་རུང་ཚང་མར་གོང་རིམ་གྱི་ཆོག་མཆན་འཐོབ་དགོས་པར་མ་ཟད། གནད་དོན་གང་ཡང་མེད་པར་གོང་རིམ་གྱི་མཛུབ་སྟོན་ལ་བརྩི་སྲུང་ཞུ་དགོས།"))
    
    # Create new tokenizer
    tokenizer = LlamaTokenizer(vocab_file=args.save_directory + 'LLM500_tokenizer.model')
    print(tokenizer)
    print(tokenizer.tokenize("Workers must often get their superiors' approval for any decisions they make, and are expected to obey their superiors' instructions without question."))
    print(tokenizer.tokenize("ངལ་རྩོལ་པར་ངེས་པར་དུ་རང་ཉིད་ཀྱིས་གཏན་འབེབས་བྱས་པའི་ཆོད་དོན་གང་ཞིག་ཡིན་རུང་ཚང་མར་གོང་རིམ་གྱི་ཆོག་མཆན་འཐོབ་དགོས་པར་མ་ཟད། གནད་དོན་གང་ཡང་མེད་པར་གོང་རིམ་གྱི་མཛུབ་སྟོན་ལ་བརྩི་སྲུང་ཞུ་དགོས།"))
    tokenizer.save_pretrained(args.save_directory + 'LLM500_tokenizer')

