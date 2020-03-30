from typing import List

from pydantic import BaseModel


# General Data Models
class Labels(BaseModel):
    value: str
    confidence: float


class Entities(BaseModel):
    text: str
    start_pos: int
    end_pos: int
    type: str
    confidence: float


class QASpanLabel(BaseModel):
    text: str
    probability: float
    start_logit: float
    end_logit: float
    start_index: int
    end_index: int


# Token Tagging Data Model
class TokenTaggingRequest(BaseModel):
    text: str


class TokenTaggingResponse(BaseModel):
    text: str
    labels: List[Labels] = []
    entities: List[Entities] = []


# Sequence Classification
class SequenceClassificationRequest(BaseModel):
    text: str


class SequenceClassificationResponse(BaseModel):
    text: str
    labels: List[Labels] = []
    entities: List[Entities] = []


# Question Answering
class QuestionAnsweringRequest(BaseModel):
    query: str
    context: str
    top_n: int = 10


class QuestionAnsweringResponse(BaseModel):
    best_answer: str
    best_n_answers: List[QASpanLabel]
