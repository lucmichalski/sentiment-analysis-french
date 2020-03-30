from adaptnlp import EasyTokenTagger
from flair.data import Sentence
from flair.data import Token


def get_tokens():
    s = Sentence()
    s.tokens = [
        Token("The", 1),
        Token("quick", 2),
        Token("brown", 3),
        Token("fox", 4),
        Token("jumps", 5),
        Token("over", 6),
        Token("the", 7),
        Token("lazy", 8),
        Token("dog", 9),
        Token(".", 10),
    ]
    s.tokenized = "The quick brown fox jumps over the lazy dog."
    return [s]


def base_case_tag_text(model):
    tagger = EasyTokenTagger()
    example_text = "The quick brown fox jumps over the lazy dog."
    sentences = tagger.tag_text(text=example_text, model_name_or_path=model)
    return sentences


def base_case_tag_all(model_list):
    tagger = EasyTokenTagger()
    example_text = "The quick brown fox jumps over the lazy dog."
    for model in model_list:
        tagger.tag_text(text=example_text, model_name_or_path=model)
    sentences = tagger.tag_all(text=example_text)
    return sentences


# tag_text method tests
def test_tag_text_type():
    sentences = base_case_tag_text("ner-fast")
    assert type(sentences) == type(get_tokens())


# tag_all method tests
def test_tag_all_type():
    sentences = base_case_tag_all(["ner-fast"])
    assert type(sentences) == type(get_tokens())
