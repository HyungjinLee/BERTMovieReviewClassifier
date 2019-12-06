#!/usr/bin/env python
# coding: utf-8

# An experiment done on 2019.11.16 by Hyungjin

# Import modules

import numpy as np
import pandas as pd
from mxnet.gluon import nn, rnn
from mxnet import gluon, autograd
import gluonnlp as nlp
from mxnet import nd
import mxnet as mx
import time
import itertools
import random
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from KaggleWord2VecUtility import KaggleWord2VecUtility

# I don't have any GPUs :) So that's why I've done these kinds of experiments on a GPU Server.

ctx = mx.gpu()

# Loading Pretrained BERT Model & Vocabulary

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                           dataset_name = 'wiki_multilingual_cased',
                                           pretrained=True,ctx=ctx,use_pooler=True,
                                           use_decoder=False, use_classifier=False)


ds = gluon.data.SimpleDataset([['BERT coding', 'not easy']])
tok = nlp.data.BERTTokenizer(vocab=vocabulary, lower= False)
trans = nlp.data.BERTSentenceTransform(tok, max_seq_length=10)
list(ds.transform(trans))

# Load Datasets in TSVData

# Train Data - 170K Plain Texts that contain polarity labels respectively. (Not Movie Reivews)
# Test Data - Kaggle Movie Review Corpora that contain polarity lables respectively.

dataset_train = nlp.data.TSVDataset("../Data/shuffled_output.txt", field_indices=[0,-1], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("../Data/KaggleTestData.txt", num_discard_samples=1)


#Getting rid of unnecessary texts in all test_data

for test_data in dataset_test :
    test_data[2] = KaggleWord2VecUtility.refine_review(test_data[2])

# Generating a BERT Tokenizer

bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower= True)
max_len = 64

# BERT Dataset

class BERTDataset(mx.gluon.data.Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        sent_dataset = gluon.data.SimpleDataset([[
            i[sent_idx],
        ] for i in dataset])
        self.sentences = sent_dataset.transform(transform)
        self.labels = gluon.data.SimpleDataset(
            [np.array(np.int32(i[label_idx])) for i in dataset])

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

data_train = BERTDataset(dataset_train, 0, 1, bert_tokenizer, max_len, True, False)
data_test = BERTDataset(dataset_test, 2, 1, bert_tokenizer, max_len+300, True, False)

# You can check the new format 
# [ text(string) -> numeric feature in the vector dimension(int), label array ] !!
# Now you are ready to utilize these as train data going to be fitted in the BERT Classifier.

# data_test.__getitem__(5000)

# BERT Classifier - The last layer where my ideas are going to be put.

class BERTClassifier(nn.Block):
    def __init__(self,
                 bert,
                 num_classes=2,
                 dropout=None,
                 prefix=None,
                 params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def forward(self, inputs, token_types, valid_length=None):
        _, pooler = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler)

# Generating a BERT Classifier and Initializing

model = BERTClassifier(bert_base,  num_classes=2, dropout=0.3)

model.classifier.initialize(ctx=ctx)
model.hybridize()

loss_function = gluon.loss.SoftmaxCELoss()

metric = mx.metric.Accuracy()

batch_size = 64
lr = 5e-5

train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=batch_size-30, num_workers=5)

# Generating a BERT Trainer

trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                       {'learning_rate' : lr, 'epsilon' : 1e-9, 'wd' : 0.01})

log_interval = 4
num_epochs = 4

# Not Applied to both layerNorm and Weight Decay

for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
    v.wd_mult = 0.0
params = [
    p for p in model.collect_params().values() if p.grad_req != 'null'
]

# Calculating learning accuracy

def evaluate_accuracy(model, data_iter, ctx=ctx) :
    acc = mx.metric.Accuracy()
    i=0
    print('start')
    
    for i, (t,v,s, label) in enumerate(data_iter) :
        print(i)
        token_ids = t.as_in_context(ctx)
        valid_length = v.as_in_context(ctx)
        segment_ids = s.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(token_ids, segment_ids, valid_length.astype('float32'))
        acc.update(preds=output, labels=label)
        i += 1
    return (acc.get()[1])

# Preparing  for the learning rate warmup

step_size = batch_size
num_train_examples = len(data_train)
num_train_steps = int(num_train_examples / step_size * num_epochs)
warmup_ratio = 0.1
num_warmup_steps = int(num_train_steps * warmup_ratio)
step_num = 0

# Training
# Get my BERT model Learned 
        
for epoch_id in range(num_epochs) :
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader) :
        print(step_num)
        step_num += 1
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else :
            offset = (step_num - num_warmup_steps) * lr / ( num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)
        with mx.autograd.record():
            # load data
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            
            #forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()
        
        #backward computation
        ls.backward()
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(token_ids.shape[0])
        
        step_loss += ls.asscalar()
        metric.update([label], [out])
        if(batch_id + 1) % (50) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.10f}, acc={:.3f}'
                         .format(epoch_id + 1, batch_id + 1, len(train_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0
            
    # Calculate Testing accuracy
    test_acc = evaluate_accuracy(model, test_dataloader, ctx)
    print('Test Acc : {}'.format(test_acc))