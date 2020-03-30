from adaptnlp import EasyQuestionAnswering


def test_question_answering():
    qa_model = EasyQuestionAnswering()
    qa_model.predict_qa(query="Test", context="Test", n_best_size=1)
