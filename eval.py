import numpy as np
import torch
import numpy as np
import argparse as ap

from model import SentimentClassifier

from data_utils import tokenization
from data_utils.preprocess import process_tweet

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
    classes = ['anger', 'no-print', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'no-print']
    result = {}
    for i, prob in enumerate(class_out[0][0]):
        result[classes[i]] = prob.item()
        
    return result

if __name__ == '__main__':
    parser: ap.ArgumentParser = ap.ArgumentParser(prog='eval.py', description='Evaluate nvidia sentiment classifier')
    parser.add_argument('-t', '--tokenizer', nargs='?', help='path to sentencepiece .model file', default='./models/ama_32k_tokenizer.model')
    parser.add_argument('-m', '--model', nargs='?', help='path to pytorch model', default='./models/optimized_model.pth')
    parser.add_argument('-e', '--example', nargs='?', help='text example to classify')
    
    args: ap.Namespace = parser.parse_args()
    model = torch.load(args.model)
    tokenizer = tokenization.SentencePieceTokenizer(model_path=args.tokenizer)
    result = eval_fn(model, tokenizer, process_tweet, args.example)
    for k, v in result.items():
        if k != 'no-print':
            print(f"{k}: {v:.4f}")
    