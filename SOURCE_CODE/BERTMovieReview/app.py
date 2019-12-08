# -*- coding: utf-8 -*- 

import flask
from flask import Flask, request, render_template
import numpy as np
from BERTClassifier import BERTClassifier
import mxnet as mx
import gluonnlp as nlp

app = Flask(__name__)

# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        if request.form['btn_identifier'] == 'predict' :
            # 업로드 리뷰 처리 
            review = request.form.get('text')
            if not review: return render_template('index.html', label="No Reviews")

            review_problem = [[review, "0"]]
        
            # 입력 받은 리뷰에서 감성 예측
            prediction = kor_model.predict(bert_base, vocabulary, ctx, review_problem)

            # 감성 표시
            if prediction == 1 :
                label = '긍정입니다.'
            elif prediction == -1 :
                label = '음... 부정도 긍정도 아닐 가능성이 높습니다.'
            else :
                label = '부정입니다.'
                
            # 결과 리턴
            return render_template('index.html', label=label)
        
        # 오류 보고 - 관리자 이형진
        elif request.form['btn_identifier'] == 'report' :
            # 업로드 리뷰 처리
            review = request.form.get('text')
            print(review)
            with open("errors.txt","a") as fo:
                fo.write(str(review)+'\n')
        
    # 결과 리턴
    return render_template('index.html')    
                    
if __name__ == '__main__':
     
    ctx = mx.gpu()

    # Loading Pretrained BERT Model & Vocabulary
    
    bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                               dataset_name = 'wiki_multilingual_cased',
                                               pretrained=True,ctx=ctx,use_pooler=True,
                                               use_decoder=False, use_classifier=False)
      
    kor_model = BERTClassifier(bert_base)
    kor_model.classifier.initialize(ctx=ctx)
    
    kor_model.load_parameters("../kor_model", ctx=ctx, allow_missing=True,ignore_extra=True)
    #kor_model.load_parameters("../mymodel_english_mid", ctx=ctx, allow_missing=True, ignore_extra=True)
    kor_model.hybridize()
    
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=163, debug=True)