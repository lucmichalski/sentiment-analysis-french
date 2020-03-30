from adaptnlp import EasyWordEmbeddings, EasyStackedEmbeddings, EasyDocumentEmbeddings


def test_easy_word_embeddings():
    embeddings = EasyWordEmbeddings()
    embeddings.embed_text(text="Test", model_name_or_path="bert-base-cased")


def test_easy_stacked_embeddings():
    embeddings = EasyStackedEmbeddings("bert-base-cased", "xlnet-base-cased")
    embeddings.embed_text(text="Test")


def test_easy_document_embeddings():
    embeddings = EasyDocumentEmbeddings("bert-base-cased", "xlnet-base-cased")
    embeddings.embed_pool(text="Test")
