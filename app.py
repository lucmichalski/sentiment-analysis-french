from summarizer import Summarizer
from flask import Flask, request

model = Summarizer()

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['POST'])
def summarizer():
	text = request.args.get('text')
	max_length = int(request.args.get('max_length'))
	min_length = int(request.args.get('min_length'))
	
	result = model(text, max_length=max_length, min_length=min_length)
	resp = ''.join(result)

	return resp

app.run(host="0.0.0.0", port=6001)

