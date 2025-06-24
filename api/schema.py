from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str

class SourceInfo(BaseModel):
    id: int
    chunk_index: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    context: str
    sources: List[SourceInfo]
    trace: List[str]