import pkg_resources

from .question_answering import BertQuestionAnsweringModel

__version__ = (
    pkg_resources.resource_string("adaptnlp", "VERSION.txt").decode("UTF-8").strip()
)

__all__ = [
    "__version__",
    "BertQuestionAnsweringModel",
]
