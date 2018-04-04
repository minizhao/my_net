import datetime
import json
import os
import shutil
import sys
import time

import torch
from tensorboard_logger import configure, log_value
from torch import optim
from torch.autograd import Variable

import models.r_net as RNet
from utils.squad_eval import evaluate
from utils.utils import make_dirs

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    checkpoint_regular = os.path.join(path, filename)
    checkpint_best = os.path.join(path, best_filename)
    torch.save(state, checkpoint_regular)
    if is_best:
        shutil.copyfile(checkpoint_regular, checkpint_best)

class Trainer(object):
    def __init__(self, args, dataloader_train, dataloader_dev,dataloader_test, word_embedding_config,
                 sentence_encoding_config, pair_encoding_config, self_matching_config, pointer_config):
        
        self.dev_dataset = args.dev_files[0]

        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        self.dataloader_test = dataloader_test

        self.model = RNet.Model(args, word_embedding_config, sentence_encoding_config,
                                pair_encoding_config, self_matching_config, pointer_config)
        self.parameters_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.Adadelta(self.parameters_trainable, rho=0.95)
        self.best_f1 = 0
        self.step = 0
        self.start_epoch = args.start_epoch
        self.name = args.name
        self.start_time = datetime.datetime.now().strftime('%b-%d_%H-%M')

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_f1 = checkpoint['best_f1']
                self.name = checkpoint['name']
                self.step = checkpoint['step']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_time = checkpoint['start_time']

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            self.name += "_" + self.start_time

        # use which device
        if torch.cuda.is_available():
            self.model = self.model.cuda(args.device_id)
        else:
            self.model = self.model.cpu()

        self.loss_fn = torch.nn.CrossEntropyLoss().cuda(args.device_id)

        configure("log/%s" % (self.name), flush_secs=5)
        self.checkpoint_path = os.path.join(args.checkpoint_path, self.name)
        make_dirs(self.checkpoint_path)

    def train(self, epoch_num):
        for epoch in range(self.start_epoch, epoch_num):
            global_loss = 0.0
            global_acc = 0.0
            last_step = self.step - 1
            last_time = time.time()

            for batch_idx, batch_train in enumerate(self.dataloader_train):
                loss, acc = self._forward(args,batch_train)
                global_loss += loss.data[0]
                global_acc += acc
                self._update_param(loss)

                if self.step % 10 == 0:
                    used_time = time.time() - last_time
                    step_num = self.step - last_step
                    print("step %d / %d of epoch %d)" % (self.step, len(self.dataloader_train), epoch), flush=True)
                    print("loss: ", global_loss / step_num, flush=True)
                    print("acc: ", global_acc / step_num, flush=True)
                    speed = self.dataloader_train.batch_size * step_num / used_time
                    print("speed: %f examples/sec \n\n" %
                          (speed), flush=True)

                    log_value('train/EM', global_acc / step_num, self.step)
                    log_value('train/loss', global_loss / step_num, self.step)
                    log_value('train/speed', speed, self.step)

                    global_loss = 0.0
                    global_acc = 0.0
                    last_step = self.step
                    last_time = time.time()
                
                if self.step % 200 == 0:
                    exact_match, f1 = self.eval(args)
                    print("exact_match: %f)" % exact_match, flush=True)
                    print("f1: %f)" % f1, flush=True)

                    log_value('dev/f1', f1, self.step)
                    log_value('dev/EM', exact_match, self.step)

                    if f1 > self.best_f1:
                        is_best = True
                        self.best_f1 = f1
                    else:
                        is_best = False

                    save_checkpoint({
                        'epoch': epoch ,
                        'state_dict': self.model.state_dict(),
                        'step': self.step ,
                        'best_f1': self.best_f1,
                        'name': self.name,
                        'optimizer': self.optimizer.state_dict(),
                        'start_time': self.start_time
                    }, is_best, self.checkpoint_path)
                self.step += 1

    def eval(self,args):
        self.model.eval()
        pred_result = {}
        for _, batch in enumerate(self.dataloader_dev):

            question_ids, questions, passages, passage_tokenized = batch
            questions.variable(args,volatile=True)
            passages.variable(args,volatile=True)
            begin_, end_ = self.model(questions, passages)  # batch x seq

            _, pred_begin = torch.max(begin_, 1)
            _, pred_end = torch.max(end_, 1)

            pred = torch.stack([pred_begin, pred_end], dim=1)

            for i, (begin, end) in enumerate(pred.cpu().data.numpy()):
                ans = passage_tokenized[i][begin:end + 1]
                qid = question_ids[i]
                pred_result[qid] = " ".join(ans)
        self.model.train()
        return evaluate(self.dev_dataset, pred_result)

    def pred(self,args):
        results=[]
        for index, batch in enumerate(self.dataloader_test):
            question_ids, questions, passages, passage_tokenized ,question_type= batch
            questions.variable(args,volatile=True)
            passages.variable(args,volatile=True)
            begin_, end_ = self.model(questions, passages)
            _, pred_begin = torch.max(begin_, 1)
            _, pred_end = torch.max(end_, 1)

            _start,_end=pred_begin.data[0],pred_end.data[0]
            if _start>_end:
                _start,_end=_end,_start
            passage_tokenized=list(passage_tokenized)[0]
            _ans=passage_tokenized[_start:_end]

            result={
              "question_id": list(question_ids)[0],
              "question_type": list(question_type)[0],
              "answers": ["".join(_ans)],
              "yesno_answers": []
            }
            results.append(result)
            if(len(results)%1000==0):
                print(index)
        assert len(results)==30000
        file = open('test_zhidao.json','w',encoding='utf-8')  
        json.dump(results,file,ensure_ascii=False)  


    def _forward(self, args,batch):

        _, questions, passages, answers, _ = batch
        batch_num = questions.tensor.size(0)

        questions.variable(args)
        passages.variable(args)

        begin_, end_ = self.model(questions, passages)  # batch x seq
        assert begin_.size(0) == batch_num

        answers = Variable(answers)
        if torch.cuda.is_available():
            answers = answers.cuda()
        begin, end = answers[:, 0], answers[:, 1]
        loss = self.loss_fn(begin_, begin) + self.loss_fn(end_, end)

        _, pred_begin = torch.max(begin_, 1)
        _, pred_end = torch.max(end_, 1)

        exact_correct_num = torch.sum(
            (pred_begin == begin) * (pred_end == end))
        em = exact_correct_num.data[0] / batch_num

        return loss, em

    def _update_param(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
