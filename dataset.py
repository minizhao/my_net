# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import sys
from torch.utils.data import Dataset
import nltk
from torch.utils.data import DataLoader
from functools import partial
import torch
from utils.utils import sort_idx
from torch.autograd import Variable

def padding(seqs, pad, batch_first=False):
    """

    :param seqs: tuple of seq_length x dim
    :return: seq_length x Batch x dim
    """
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()
    return (seq_tensor, lengths)


class Documents(object):
    """
        Helper class for organizing and sorting seqs

        should be batch_first for embedding
    """
    def __init__(self, tensor, lengths):
        self.original_lengths = lengths
        sorted_lengths_tensor, self.sorted_idx = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)

        self.tensor = tensor.index_select(dim=0, index=self.sorted_idx)

        self.lengths = list(sorted_lengths_tensor)
        self.original_idx = torch.LongTensor(sort_idx(self.sorted_idx))

        self.mask_original = torch.zeros(*self.tensor.size())
        
        for i, length in enumerate(self.original_lengths):
            self.mask_original[i][:length].fill_(1)

    def variable(self,args, volatile=False):
        self.tensor = Variable(self.tensor, volatile=volatile).cuda(args.device_id)
        self.sorted_idx = Variable(self.sorted_idx, volatile=volatile).cuda(args.device_id)
        self.original_idx = Variable(self.original_idx, volatile=volatile).cuda(args.device_id)
        self.mask_original = Variable(self.mask_original, volatile=volatile).cuda(args.device_id)
        return self

    def restore_original_order(self, sorted_tensor, batch_dim):
        return sorted_tensor.index_select(dim=batch_dim, index=self.original_idx)

    def to_sorted_order(self, original_tensor, batch_dim):
        return original_tensor.index_select(dim=batch_dim, index=self.sorted_idx)


class RawExample(object):
    pass            

def get_passage_ans(sample):
    question_tokens = sample['segmented_question']
    passages_tokens=[]
    answer_docs=sample["answer_docs"][0]  #第几个答案所在的位置
    offset=0
    
    for d_idx, doc in enumerate(sample['documents']):
        #每个sample 即每一行的记录的‘documents’是有多个文章的list 5个
        """
        每一个doc里面的内容：[u'is_selected', u'title', u'most_related_para',
        u'segmented_title', u'segmented_paragraphs', u'paragraphs', u'bs_rank_pos']

        """
        if d_idx==answer_docs:
            offset=len(passages_tokens) #计算答案位置的偏移量
        most_related_para = doc['most_related_para']
        passages_tokens.extend(doc['segmented_paragraphs'][most_related_para])
   
    answers_tokens= sample['fake_answers']
    answer_docs=sample["answer_docs"][0]
    #[x+offset for x in sample["answer_spans"]]
    answer_spans=sample["answer_spans"][0]

    answer_spans=[x+offset for x in answer_spans]
    
    return question_tokens,passages_tokens,answers_tokens,answer_spans

    
#读取json文件            
def read_train_json(path, debug_mode, debug_len, delete_long_context=True, delete_long_question=True,
                    longest_context=300, longest_question=30):
    
    examples = []
    with open(path) as fin:
        for lidx, line in enumerate(fin):
            #取得一行的样本
            sample = json.loads(line.strip())
            
            if len(sample['answer_spans']) == 0 or len(sample['answer_spans'][0])!=2:
                continue
                    
            title=([x['segmented_title'] for x in sample['documents']])
            question_tokens,passages_tokens,answers_tokens,answer_spans=get_passage_ans(sample)

            if len(passages_tokens)>500:
                continue
            question_id = sample["question_id"]
            e = RawExample()
            e.title = title
            e.tokenized_passage = passages_tokens
            e.tokenized_question = question_tokens
            e.question_id = question_id
            e.answer_position=(answer_spans[0],answer_spans[1])
            e.answer_text = answers_tokens
            examples.append(e)

    return examples


def read_test_json(path, debug_mode, debug_len, delete_long_context=True, delete_long_question=True,
                    longest_context=300, longest_question=30):
    examples = []
    with open(path) as fin:
        for lidx, line in enumerate(fin):
            #取得一行的样本
            sample = json.loads(line.strip())
            documents=sample["documents"] #one sample has may doucuments
            #paragraphs
            str_=""
            for i in range(len(documents)):
                for x in documents[0]['segmented_paragraphs']:
                    str_+=" ".join(x)+" " 
            
            passages_tokens=str_.split(" ")
            if len(passages_tokens)>500:
                passages_tokens=passages_tokens[:500]
            e = RawExample()
            e.tokenized_passage = passages_tokens
            e.tokenized_question = sample["segmented_question"]
            e.question_id = sample["question_id"]
            e.question_type=sample["question_type"]
            examples.append(e)
    return examples
    

            
class SQuAD(Dataset):
    def __init__(self, path, itos, stoi, tokenizer="nltk", split="train",
                 debug_mode=False, debug_len=50):

        self.insert_start = stoi.get("<SOS>", None)
        self.insert_end = stoi.get("<EOS>", None)
        self.UNK = stoi.get("<UNK>", None)
        self.PAD = stoi.get("<PAD>", None)
        self.stoi = stoi
        self.itos = itos
        self.split = split

        # Read and parsing raw data from json
        # Tokenizing with answer may result in different tokenized passage even the passage is the same one.
        # So we tokenize passage for each question in train split
        if self.split == "train":
            self.examples = read_train_json(path, debug_mode, debug_len)
            #找到开始词和结尾词
        else:
            self.examples = read_test_json(path, debug_mode, debug_len)
          
        for e in self.examples:
            #对每一个样本来处理
            e.numeralized_question = self._numeralize_word_seq(e.tokenized_question, self.stoi)
            e.numeralized_passage = self._numeralize_word_seq(e.tokenized_passage, self.stoi)


    def _numeralize_word_seq(self, seq, stoi, insert_sos=False, insert_eos=False):
        if self.insert_start is not None and insert_sos:
            result = [self.insert_start]
        else:
            result = []
        for word in seq:
            result.append(stoi.get(word, 0))
        if self.insert_end is not None and insert_eos:
            result.append(self.insert_end)
        return result

    def __getitem__(self, idx):
        item = self.examples[idx]
        if self.split == "train":
            return (item, item.question_id, item.numeralized_question, 
                    item.numeralized_passage, 
                    item.answer_position, item.answer_text, item.tokenized_passage)
        else:
            return (item, item.question_id, item.question_type, item.numeralized_question, 
                    item.numeralized_passage, 
                    item.tokenized_passage)

    def __len__(self):
        return len(self.examples)
   
    def _create_collate_fn(self, batch_first=True):

        def collate(examples, this):
            if this.split == "train":
                items, question_ids, questions, passages, answers_positions, answer_texts, passage_tokenized = zip(
                    *examples)
            else:
                items, question_ids, question_type, questions, passages, passage_tokenized = zip(*examples)

            questions_tensor, question_lengths = padding(questions, this.PAD, batch_first=batch_first)
            passages_tensor, passage_lengths = padding(passages, this.PAD, batch_first=batch_first)

            # TODO: implement char level embedding

            question_document = Documents(questions_tensor, question_lengths)
            passages_document = Documents(passages_tensor, passage_lengths)
            


            if this.split == "train":
                return question_ids, question_document, passages_document, torch.LongTensor(answers_positions), answer_texts

            else:
                return question_ids, question_document, passages_document, passage_tokenized,question_type

        return partial(collate, this=self)

    def get_dataloader(self, batch_size, num_workers=4, shuffle=True, batch_first=True, pin_memory=False):
        """

        :param batch_first:  Currently, it must be True as nn.Embedding requires batch_first input
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, drop_last=True,pin_memory=pin_memory)
            
