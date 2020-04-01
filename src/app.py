import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
import traceback

import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0" 

from transformers import CamembertTokenizer, TFCamembertForSequenceClassification

try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus

app = Flask(__name__)

@app.route("/process", methods=["POST", "GET"])
def process():
    try:
        text = request.json["text"]

        print("text:", text)

        # Inference
        MAX_SEQ_LEN = 400
        # text = "Ce film était génial !"
        X = encode_reviews(tokenizer, [text], MAX_SEQ_LEN)
        scores = model.predict(X)
	
        # print(type(scores))
        # print(scores)
        # print(scores[0])
        # print(scores[0].shape)

        y_pred = np.argmax(scores[0])
        # y_pred = np.argmax(scores[0], axis=1)
        # y_pred = 0 if negative, 1 if positive
        # here, y_pred shoud be 1 
        # print("y_pred:", y_pred)

        if y_pred == 1:
           prediction = "Positive"
        else:    
           prediction = "Negative"

        callback = json.dumps({"y_pred": int(y_pred), "prediction": prediction})
        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

# Preprocessing 
def encode_reviews(tokenizer, reviews, max_length):
    token_ids = np.zeros(shape=(len(reviews), max_length),
                         dtype=np.int32)
    for i, review in enumerate(reviews):
        encoded = tokenizer.encode(review, max_length=max_length)
        token_ids[i, 0:len(encoded)] = encoded
    attention_mask = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_mask": attention_mask}

if __name__ == '__main__':
    global model, tokenizer

    # Load model 
    MODEL_FOLDER = "../models/camembert_sentiment" # Local model
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = TFCamembertForSequenceClassification.from_pretrained(MODEL_FOLDER)

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=False)  

