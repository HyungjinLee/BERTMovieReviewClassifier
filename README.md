These are a series of experiments for KHU Capstone Design Project 2019.

Such a shame... Environments where I have been doing experiments are completely different. 

# Deep Learning Environments
    
    BERT (My Model) : MXNET
    Text_Claasification_CNN, Bi_LSTM : Tensorflow 

So I'll share all the GitHub links respectively. Please refer to the links on folders. From this way, I think it is gonna be much easier to execute them.

I don't have any GPUs :) So that's why I've done these kinds of experiments on a GPU Server. If you don't, you might be working on it with your CPU which will take so long time... (A day per one epoch) I highly recommend using GPUs.

# English Datasets
    
     Train Data - 170K Plain Texts that contain polarity labels respectively. (Not Movie Reivews, a sentence per data)
     Test Data - 25K Kaggle Movie Review Corpora that contain polarity lables respectively. (11~17 sentences per data on Average)

# Korean Datasets

     Train Data - 150K Plain Texts that contain polarity labels crawled from Naver Movie.
     Test Data - 50K Plain Texts that contain polarity labels crawled from Naver Movie.
 	 
 	 
# Results

 	Trial	Representation Model	                        Train Data	                                Test Data	    Last Layer Classifier	Test Accuracy(%)
        
        1   wiki_multilingual_cased (Google) + Fine Tuning	NAVER Movie Reviews(Korean)	      NAVER Movie Reviews(Korean)	  Softmax Layer	        *87.1
        
        2   wiki_multilingual_cased (Google) + Fine Tuning	  170K Plain Texts(English)	25K Kaggle Movie Corpora(English)	  Softmax Layer	    *82.1 -> 83.03	
        
        3   korean_bert_model(ETRI) + Fine Tuning	        NAVER Movie Reviews(Korean)	      NAVER Movie Reviews(Korean)	  Softmax Layer	
        
        4	Word2Vec (Google) + Word Embedding	              170K Plain Texts(English)	25K Kaggle Movie Corpora(English)	  CNN Layer	            68.0
        
        5	Word2Vec (Google) + Word Embedding	            NAVER Movie Reviews(Korean)	      NAVER Movie Reviews(Korean) Bi-LSTM Softmax Layer	    86.5
					
