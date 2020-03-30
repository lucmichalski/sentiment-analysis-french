from nboost.plugins.models import resolve_model
from nboost import defaults
import unittest


class TestTfBertModel(unittest.TestCase):

    def setUp(self):
        self.model = resolve_model(
            model_dir='tf-bert-base-uncased-msmarco',
            data_dir=defaults.data_dir,
            model_cls=''
        )

    # def test_rank(self):
    #     ranks = self.model.rank('O wherefore art thou', CHOICES)
    #     self.assertIsInstance(ranks, list)
    #     self.assertEqual(6, len(ranks))

    def tearDown(self) -> None:
        self.model.close()


CHOICES = [
    'From fairest creatures we desire increase',
    'That thereby beautys rose might never die',
    'But as the riper should by time decease',
    'His tender heir might bear his memory:',
    'But thou contracted to thine own bright eyes',
    'Feedst thy lights flame with self-substantial fuel',
]