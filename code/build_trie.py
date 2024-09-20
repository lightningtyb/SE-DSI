#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
from transformers import BertTokenizer, BartTokenizer, T5Tokenizer
from tqdm import tqdm
import pickle
import json
import jsonlines
import os
sys.path.append(r"/data/users/tangyubao/DSI")
from genre.trie import Trie
from genre.hf_model import GENRE




def load_lst(path):
    lst = []
    with open(path, 'r') as f :
        for line in tqdm(f,desc='loading'):
            lst.append(line.rstrip())
    return lst


def load_json(path,sub_lst=None):
    lst = []
    with jsonlines.open(path) as reader:
        for line in tqdm(reader,desc='loading'):
            lst.append(line['translation']['tgt'])
    if sub_lst:
        lst.extend(sub_lst)
    return lst


def build_hash_tree(all_title, save_path):
    candidates_trie = Trie()
    for t in tqdm(all_title, desc='building...'):

        candidates_trie.add(t)
        # print(t)
        # print(candidates_trie.trie_dict)

    print('writing tree...')
    with open(save_path, 'wb') as f:
        pickle.dump(candidates_trie, f)

    print('trie len:', candidates_trie.len)


def encode_hash(hash, tokenizer, start_id, save_path):
    labels = tokenizer(hash)
    decoder_input = []
    input_ids = labels['input_ids']

    i = 0
    for idx, l in enumerate(input_ids):
    # for l in input_ids:
        l.insert(0, start_id)
        # print(f"{hash[idx]}--{l}")
        # if i >5:
        #     break
        # i += 1
        decoder_input.append(l)
    build_hash_tree(decoder_input, save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--allid", type=str, default="all id path")
    parser.add_argument("--model", type=str, default="model path")
    parser.add_argument("--save", type=str, default="save pkl path")


    args = parser.parse_args()
    assert os.path.exists(args.allid)

    hash = load_lst(args.allid)
    t5_tok = T5Tokenizer.from_pretrained(args.model)
    encode_hash(hash, t5_tok, t5_tok.pad_token_id,args.save)






