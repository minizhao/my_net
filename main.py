import config
import sys
import os
import pickle
import torch
import h5py
import tqdm
import json
from itertools import chain
from trainer import Trainer
from dataset import SQuAD

def get_words(args,min_count=5):

    words_sava_file=os.path.join(args.vocab_dir, "news12g_bdbk20g_nov90g_dim128","words.pkl")
    if os.path.isfile(words_sava_file):
        words_file = open(words_sava_file, 'rb')
        words = pickle.load(words_file)
        return words

    
    #得到全部数据集的词汇表
    files=[args.train_files[0],args.dev_files[0]]
    vocab = {}
    for f in files:
        with open(f, 'r') as fin:
            for line in fin:
                obj = json.loads(line.strip())
                paras = [
                        chain(*d['segmented_paragraphs'])
                        for d in obj['documents']]
                doc_tokens = chain(*paras)
                question_tokens = obj['segmented_question']
                for t in list(doc_tokens) + question_tokens:
                    vocab[t] = vocab.get(t, 0) + 1
    # output
    sorted_vocab = sorted([(v, c) for v, c in vocab.items()],
            key=lambda x: x[1],
            reverse=True)
    words=[x[0] for x in sorted_vocab if x[1]>min_count]
    
    words_file = open(words_sava_file, 'wb')
    pickle.dump(words,words_file)

    return words

def read_vocab(args,pretrain_wv=False):
    """
    加载词向量数据
    """
    vocab_config = {
        "<UNK>": 0,
        "<PAD>": 1,
        "<start>": 2,
        "<end>": 3,
        "insert_start": "<SOS>",
        "insert_end": "<EOS>",
        "tokenization": "nltk",
        "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
        "embedding_path": os.path.join(args.vocab_dir, "news12g_bdbk20g_nov90g_dim128", "news12g_bdbk20g_nov90g_dim128.bin"),
        "embedding_dim": 64
    }
    filename=os.path.join(args.vocab_dir, "news12g_bdbk20g_nov90g_dim128","to_torch.h5")
    
    #是否有之前保存好的词向量数据
    if os.path.isfile(filename):
        with h5py.File(filename, 'r') as h:
            itos=h['itos']
            stoi=h['stoi']
            vectors=h['vectors']
        return itos, stoi, vectors
    
    #wv_dict 是导入外部词嵌入的vocab，type：list
    embed_size = vocab_config["embedding_dim"]
    #词嵌入维度64
    print("word embedding size: %d" % embed_size)
    words=get_words(args)
    print("words count size: %d" % len(words))

    #先把特殊字符加上list里面
    itos = vocab_config['specials'][:]
    stoi = {}
    itos.extend(words)
    stoi=dict(zip(itos,range(len(itos))))
    vectors = torch.zeros([len(itos), embed_size])
    
    if pretrain_wv:
        wv_dict, wv_vectors, wv_size = read_embedding(vocab_config["embedding_path"],
                                                  vocab_config["embedding_dim"])
        for word, idx in tqdm.tqdm(stoi.items()):
            if word not in wv_dict or word in vocab_config['specials']:
                continue
            vectors[idx, :wv_size]=torch.from_numpy(wv_vectors[word])
        #释放词嵌入空间
        del wv_vectors
    
   
    return itos, stoi, vectors


def read_dataset(json_file, itos, stoi, is_debug=False, split="train"):
    
    dataset = SQuAD(json_file, itos, stoi, debug_mode=is_debug, split=split)
    return dataset


def train():
    args = config.parse_args()
    
    #加载词嵌入向量数据
    itos, stoi, vectors=read_vocab(args)
    
    word_embedding_config = {"embedding_weights": vectors,
                             "padding_idx": 0,
                             "update": True}

    sentence_encoding_config = {"hidden_size": args.hidden_size,
                                "num_layers": args.num_layers,
                                "bidirectional": True,
                                "dropout": args.dropout, }

    pair_encoding_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": args.bidirectional,
                            "dropout": args.dropout,
                            "gated": True, "mode": "GRU",
                            "rnn_cell": torch.nn.GRUCell,
                            "attn_size": args.attention_size,
                            "residual": args.residual}

    self_matching_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": args.bidirectional,
                            "dropout": args.dropout,
                            "gated": True, "mode": "GRU",
                            "rnn_cell": torch.nn.GRUCell,
                            "attn_size": args.attention_size,
                            "residual": args.residual}

    pointer_config = {"hidden_size": args.hidden_size,
                      "num_layers": args.num_layers,
                      "dropout": args.dropout,
                      "residual": args.residual,
                      "rnn_cell": torch.nn.GRUCell}
    
    #load the dataset
    train_json = args.train_files[0]
    dev_json = args.dev_files[0]
    test_json = args.test_files[0]
    
    test = read_dataset(test_json, itos, stoi,split="test")
    train = read_dataset(train_json, itos, stoi,split="train")
    dev = read_dataset(dev_json, itos, stoi, split="train")


    dev_dataloader = dev.get_dataloader(args.batch_size)
    train_dataloader = train.get_dataloader(args.batch_size, shuffle=True)
    test_dataloader = test.get_dataloader(args.test_batch_size)

    
    trainer = Trainer(args, train_dataloader, dev_dataloader,test_dataloader,
                       word_embedding_config,
                      sentence_encoding_config, pair_encoding_config,
                      self_matching_config, pointer_config)
    # print(len(test_dataloader))
    # trainer.pred()


if __name__ == '__main__':
    train()        
