from adaptnlp import SequenceClassifierTrainer, EasyDocumentEmbeddings
from flair.datasets import TREC_6


def test_sequence_classifier_trainer():
    corpus = TREC_6()

    # Instantiate AdaptNLP easy document embeddings module, which can take in a variable number of embeddings to make `Stacked Embeddings`.
    # You may also use custom Transformers LM models by specifying the path the the language model
    doc_embeddings = EasyDocumentEmbeddings("bert-base-cased", methods=["rnn"])

    # Instantiate Sequence Classifier Trainer by loading in the data, data column map, and embeddings as an encoder
    trainer = SequenceClassifierTrainer(
        corpus=corpus, encoder=doc_embeddings, column_name_map={0: "text", 1: "label"}
    )

    trainer
