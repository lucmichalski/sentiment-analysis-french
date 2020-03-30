"""
This code is based on implementation from transformers library https://github.com/huggingface/transformers/blob/5daca95dddf940139d749b1ca42c59ebc5191979/src/transformers/convert_bert_pytorch_checkpoint_to_original_tf.py
But this particular code snippet implements conversion of BertForSequenceClassification model which isn't supported by transformers library as of now.
A sample tf_bert_config_file json file can be found in this gist https://gist.github.com/artoby/b13d2b9d2d6d7f21e195bdb8542709c6

"""

import os
import tensorflow as tf
from transformers import (
    BertForSequenceClassification,
)
import numpy as np
import json

def bert_pytorch_to_tensorflow(pt_model:BertForSequenceClassification,
                               tf_bert_config_file:str,
                               tf_output_dir:str,
                               tf_model_name:str = "bert_model"):

    """
    Converts a PyTorch transformers BertForSequenceClassification model to Tensorflow
    :param pt_model: PyTorch model instance to be converted
    :param tf_bert_config_file: path to bert_config.json file with Tensorflow BERT configuration.
      This config file should correspond to the architecture (N layers, N hidden units, etc.) of the PyTorch model. 
      Hopefully in future the code below will be improved and this config file will be generated on the fly. 
      Feel free to contribute such an implementation to https://gist.github.com/artoby/b13d2b9d2d6d7f21e195bdb8542709c6
    :param tf_output_dir: directory to write resulting Tensorflow model to
    :param tf_model_name: resulting Tensorflow model name (will be used in a file name)
    :return:
    """

    tensors_to_transpose = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    # Pytorch name, TF name, continue if found
    name_patterns_map = (
        ('classifier.weight', 'output_weights', False),
        ('classifier.bias', 'output_bias', False),
        ('layer.', 'layer_', True),
        ('word_embeddings.weight', 'word_embeddings', True),
        ('position_embeddings.weight', 'position_embeddings', True),
        ('token_type_embeddings.weight', 'token_type_embeddings', True),
        ('.', '/', True),
        ('LayerNorm/weight', 'LayerNorm/gamma', True),
        ('LayerNorm/bias', 'LayerNorm/beta', True),
        ('weight', 'kernel', True)
    )

    if not os.path.isdir(tf_output_dir):
        os.makedirs(tf_output_dir)

    state_dict = pt_model.state_dict()

    def to_tf_var_name(name:str):
        for patt, repl, cont in iter(name_patterns_map):
            if patt in name:
                name = name.replace(patt, repl)
                if not cont:
                    break
        return name

    def create_tf_var(tensor:np.ndarray, name:str, session:tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    
    all_vars = {}
    
    pytorch_vars = dict([(k, v.numpy()) for (k, v) in state_dict.items()])
    all_vars.update(pytorch_vars)
    
    additional_vars = {"global_step": np.array([0])}
    all_vars.update(additional_vars)
    
    with tf.Session() as session:
        for var_name, np_value in all_vars.items():
            print(var_name)
            tf_name = to_tf_var_name(var_name)
            if any([x in var_name for x in tensors_to_transpose]):
                np_value = np_value.T
            tf_var = create_tf_var(tensor=np_value, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, np_value)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, np_value)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(tf_output_dir, tf_model_name.replace("-", "_") + ".ckpt"))
    
    with open(tf_bert_config_file) as f:
        tf_bert_config = json.load(f)
    
    vocab_size = pytorch_vars["bert.embeddings.word_embeddings.weight"].shape[0]
    tf_bert_config["vocab_size"] = vocab_size
    with open(os.path.join(tf_output_dir, "bert_config.json"), "w") as f:
        json.dump(tf_bert_config, f, indent=2)
