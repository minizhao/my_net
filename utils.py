import array
import errno
import json
import os
import random
import zipfile
from argparse import ArgumentParser
from collections import Counter
from os.path import dirname, abspath

import nltk
import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import trange, tqdm
from gensim.models import Word2Vec,KeyedVectors

URL = {
    'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip'
}



def get_args():
    parser = ArgumentParser(description='PyTorch R-net')

    parser.add_argument('--name', type=str, default="r-net")
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--batch_size_dev', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint")
    parser.add_argument('--resume', type=str, default='/home/lab713/data1/git_code/R-net/checkpoint/r-net_Mar-01_11-15/checkpoint.pth.tar')
    parser.add_argument('--train_json', type=str, default="./data/squad/train-v1.1.json")
    parser.add_argument('--dev_json', type=str, default="./data/squad/dev-v1.1.json")
    parser.add_argument('--update_word_embedding', type=bool, default=False)
    parser.add_argument('--update_char_embedding', type=bool, default=True)
    parser.add_argument('--hidden_size', type=int, default=75)
    parser.add_argument('--attention_size', type=int, default=75)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--app_path', type=str, default=dirname(dirname(abspath(__file__))))
    parser.add_argument('--pin_memory', type=bool, default=False)

    args = parser.parse_args()
    return args


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner



class RawExample(object):
    pass


def make_dirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(url, download_path, filename):
    if not os.path.exists(os.path.join(download_path, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url, os.path.join(download_path, filename), reporthook=t.update_to)
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e


def read_train_json(path, debug_mode, debug_len, delete_long_context=True, delete_long_question=True,
                    longest_context=300, longest_question=30):
    with open(path) as fin:
        data = json.load(fin)
    examples = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            passage = p['context']
            if delete_long_context and len(nltk.word_tokenize(passage)) > longest_context:
                continue
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]

                if delete_long_question and len(nltk.word_tokenize(question)) > longest_question:
                    continue

                question_id = qa["id"]
                for ans in answers:
                    answer_start = int(ans["answer_start"])
                    answer_text = ans["text"]
                    e = RawExample()
                    e.title = title
                    e.passage = passage
                    e.question = question
                    e.question_id = question_id
                    e.answer_start = answer_start
                    e.answer_text = answer_text
                    examples.append(e)

                    if debug_mode and len(examples) >= debug_len:
                        return examples
    print("train examples :%s" % len(examples))
    return examples


def get_counter(*seqs):
    word_counter = {}
    char_counter = {}
    for seq in seqs:
        for doc in seq:
            for word in doc:
                word_counter.setdefault(word, 0)
                word_counter[word] += 1
                for char in word:
                    char_counter.setdefault(char, 0)
                    char_counter[char] += 1
    return word_counter, char_counter


def read_dev_json(path, debug_mode, debug_len):
    with open(path) as fin:
        data = json.load(fin)
    examples = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']

            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                question_id = qa["id"]
                answer_start_list = [ans["answer_start"] for ans in answers]
                c = Counter(answer_start_list)
                most_common_answer, freq = c.most_common()[0]
                answer_text = None
                answer_start = None
                if freq > 1:
                    for i, ans_start in enumerate(answer_start_list):
                        if ans_start == most_common_answer:
                            answer_text = answers[i]["text"]
                            answer_start = answers[i]["answer_start"]
                            break
                else:
                    answer_text = answers[random.choice(range(len(answers)))]["text"]
                    answer_start = answers[random.choice(range(len(answers)))]["answer_start"]

                e = RawExample()
                e.title = title
                e.passage = context
                e.question = question
                e.question_id = question_id
                e.answer_start = answer_start
                e.answer_text = answer_text
                examples.append(e)

                if debug_mode and len(examples) >= debug_len:
                    return examples

    return examples


def tokenized_by_answer(context, answer_text, answer_start, tokenizer):
    """
    Locate the answer token-level position after tokenizing as the original location is based on
    char-level

    snippet modified from: https://github.com/haichao592/squad-tf/blob/master/dataset.py

    :param context:  passage
    :param answer_text:     context/passage
    :param answer_start:    answer start position (char level)
    :param tokenizer: tokenize function
    :return: tokenized passage, answer start index, answer end index (inclusive)
    """
    fore = context[:answer_start]
    mid = context[answer_start: answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]

    tokenized_fore = tokenizer(fore)
    tokenized_mid = tokenizer(mid)
    tokenized_after = tokenizer(after)
    tokenized_text = tokenizer(answer_text)

    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None

    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token


def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}


def read_embedding(path, dim):
    
    #wv_dict, wv_vectors, wv_size = load_word_vectors(root, word_type, dim)
    model =KeyedVectors.load_word2vec_format(path,binary=True)
    return list(model.wv.vocab), model.wv, dim


def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        network = torch.nn.GRU
    elif rnn_type == "lstm":
        network = torch.nn.LSTM
    else:
        raise ValueError("Invalid RNN type %s" % rnn_type)
    return network


def sort_idx(seq):
    """

    :param seq: variable
    :return:
    """
    return sorted(range(seq.size(0)), key=lambda x: seq[x])


def prepare_data():
    make_dirs("data/cache")
    make_dirs("data/embedding/char")
    make_dirs("data/embedding/word")
    make_dirs("data/squad")
    make_dirs("data/trained_model")
    make_dirs("checkpoint")

    nltk.download("punkt")

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"
    squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

    train_url = os.path.join(squad_base_url, train_filename)
    dev_url = os.path.join(squad_base_url, dev_filename)

    download_prefix = os.path.join("data", "squad")
    maybe_download(train_url, download_prefix, train_filename)
    maybe_download(dev_url, download_prefix, dev_filename)

    char_embedding_pretrain_url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt"
    char_embedding_filename = "glove_char.840B.300d.txt"
    maybe_download(char_embedding_pretrain_url, "data/embedding/char", char_embedding_filename)
