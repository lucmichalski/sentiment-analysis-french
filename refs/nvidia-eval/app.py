import numpy as np
import torch
import numpy as np
import argparse as ap

from model import SentimentClassifier

from data_utils import tokenization
from data_utils.preprocess import process_tweet
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

torch.nn.functional.tanh = torch.tanh

def eval_fn(model, tokenizer, preprocess_fn, text):
    ids = tokenizer.EncodeAsIds(text, preprocess_fn).tokenization
    length = len(ids)
    
    input_text = torch.from_numpy(np.array(ids, dtype=np.int64))
    input_text = torch.unsqueeze(input_text, -1)
    
    input_tsteps = torch.from_numpy(np.array(length-1, dtype=np.int64))
    input_tsteps = torch.unsqueeze(input_tsteps, -1)
    
    with torch.no_grad():
        class_out = model(input_text, input_tsteps)
    classes = ['anger', 'no-print', 'disgust', 'fear', 'happiness', 'sadness', 'surprise',
            'no-print']
    result = {}
    for i, prob in enumerate(class_out[0][0]):
        result[classes[i]] = prob.item()
        
    return result


model = torch.load('./models/optimized_model.pth')
tokenizer = tokenization.SentencePieceTokenizer(
        model_path='./models/ama_32k_tokenizer.model')
    

def nvidia_eval(query):
    emotions = {}
    if query:
        result = eval_fn(model, tokenizer, process_tweet, query)
        for k, v in result.items():
            if k != 'no-print':
                emotions[k] = v
    return emotions


@app.route("/nvidia")
def nvidia():
    query = request.args.get('query', '').strip()
    emotions = nvidia_eval(query)
    emotions_str = '\n'.join(map(lambda elem: '%s: %.4f' % elem, 
        emotions.items()))

    return render_template('nvidia.html', query=query, emotions=emotions_str)


@app.route("/api/v1/nvidia", methods=['POST'])
def nvidia_api():
    query = request.form.get('query', '').strip()
    emotions = nvidia_eval(query)

    return jsonify({'query': query, 'emotions': emotions})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
