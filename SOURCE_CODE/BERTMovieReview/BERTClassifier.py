# -*- coding: utf-8 -*- 

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
from BERTDataset import BERTDataset 

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
        
    # Make a prediction 
    def predict(self, bert, vocabulary, ctx, problem) :
        
        bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower = False)
        problem = BERTDataset(problem, 0, 1, bert_tokenizer, 64, True, False)
        my_test_dataloader = mx.gluon.data.DataLoader(problem, batch_size = 1, num_workers=1)
        
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(my_test_dataloader) :
   
            with mx.autograd.record():
                        # load data
                        token_ids = token_ids.as_in_context(ctx)
                        valid_length = valid_length.as_in_context(ctx)
                        segment_ids = segment_ids.as_in_context(ctx)
                        label = label.as_in_context(ctx)
            
                        #forward computation
                        out = self(token_ids, segment_ids, valid_length.astype('float32'))
                        print(out)
    
        # return an answer that model derives
        return out.argmax(axis = 1)     
        

    def forward(self, inputs, token_types, valid_length=None):
        _, pooler = self.bert(inputs, token_types, valid_length)

        return self.classifier(pooler)