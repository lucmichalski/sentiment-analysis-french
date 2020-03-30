import os
import sys
import logging

from flask import Flask, request
from flask_json import FlaskJSON, JsonError, as_json

import worker_config
from worker.bert_feature_extractor import BERTFeatureExtractor


def create_app():

    worker_app = Flask(__name__)
    FlaskJSON(worker_app)

    pid = os.getpid()

    extractor = BERTFeatureExtractor(
        worker_config.graph_path,
        worker_config.vocab_path,
        use_gpu=worker_config.use_gpu,
        batch_size=worker_config.batch_size,
        seq_len=worker_config.seq_len)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    @worker_app.route('/encode', methods=['POST'])
    @as_json
    def encode_query():
        data = request.form if request.form else request.json
        try:
            logger.info('new request from {} to {}'.format(request.remote_addr, pid))
            encoded = extractor(data['texts']).tolist()
            payload = {'result': encoded}
            return payload

        except Exception as e:
            logger.error('error when handling HTTP request', exc_info=True)
            raise JsonError(description=str(e), type=str(type(e).__name__))

    return worker_app


app = create_app()
