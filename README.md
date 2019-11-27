These are a series of experiments for KHU Capstone Design Project 2019.

Such a shame... Environments where I have been doing experiments are completely different. 

# Deep Learning Environments
    
    BERT (My Model) : MXNET
    Text_Claasification_CNN, Bi_LSTM : Tensorflow 

So I'll share all the GitHub links respectively. Please refer to the links on folders. From this way, I think it is gonna be much easier to execute them.

I don't have any GPUs :) So that's why I've done these kinds of experiments on a GPU Server. If you don't, you might be working on it with your CPU which will take so long time... (A day per one epoch!) I highly recommend using GPUs.

# English Datasets
    
     Train Data - 170K Plain Texts that contain polarity labels respectively. (Not Movie Reivews, a sentence per data)
     Test Data - 25K Kaggle Movie Review Corpora that contain polarity lables respectively. (11~17 sentences per data on Average)

# Korean Datasets

     Train Data - 150K Plain Texts that contain polarity labels crawled from Naver Movie.
     Test Data - 50K Plain Texts that contain polarity labels crawled from Naver Movie.
 	 
 	 
# Results

    | Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `network`                     | The network to train, which is defined in [symbol/](https://github.com/dmlc/mxnet/tree/master/example/image-classification/symbols). Some networks may accept additional arguments, such as `--num-layers` is used to specify the number of layers in ResNet. |
| `data-train`, `data-val`      | The data for training and validation. It can be either a filename or a directory. For the latter, all files in the directory will be used. But if `--benchmark 1` is used, then there two arguments will be ignored. |
| `gpus`                        | The list of GPUs to use, such as `0` or `0,3,4,7`. If an empty string `''` is given, then we will use CPU. |
| `batch-size`                  | The batch size for SGD training. It specifies the number of examples used for each SGD iteration. If we use *k* GPUs, then each GPU will compute *batch_size/k* examples in each time. |
| `model`                       | The model name to save (and load). A model will be saved into two parts: `model-symbol.json` for the network definition and `model-n.params` for the parameters saved on epoch *n*. |
| `num-epochs`                  | The maximal number of epochs to train.   |
| `load-epoch`                  | If given integer *k*, then resume the training starting from epoch *k* with the model saved at the end of epoch *k-1*. Note that the training starts from epoch 0, and the model saved at the end of this epoch will be `model-0001.params`. |
| `lr`                          | The initial learning rate, namely for epoch 0. |
| `lr-factor`, `lr-step-epochs` | Reduce the learning rate on give epochs. For example, `--lr-factor .1 --lr-step-epochs 30,60` will reduce the learning rate by 0.1 on epoch 30, and then reduce it by 0.1 again on epoch 60. |

    | Trial |	Representation Model	                   |      Train Data	        |             Test Data	           |Last Layer Classifier|	Test Accuracy(%) |
    |_______|______________________________________________|____________________________|__________________________________|_____________________|___________________|    
    |   1   |   wiki_multilingual_cased (Google) + Fine Tuning	NAVER Movie Reviews(Korean)	      NAVER Movie Reviews(Korean)	  Softmax Layer	            *87.1
    |    
        2   |wiki_multilingual_cased (Google) + Fine Tuning	  170K Plain Texts(English)	25K Kaggle Movie Corpora(English)	  Softmax Layer	    *82.1 -> 83.03	
        
        3   korean_bert_model(ETRI) + Fine Tuning	        NAVER Movie Reviews(Korean)	      NAVER Movie Reviews(Korean)	  Softmax Layer	
        
        4	Word2Vec (Google) + Word Embedding	              170K Plain Texts(English)	25K Kaggle Movie Corpora(English)	  CNN Layer	            68.0
        
        5	Word2Vec (Google) + Word Embedding	            NAVER Movie Reviews(Korean)	      NAVER Movie Reviews(Korean) Bi-LSTM Softmax Layer	    86.5
					
