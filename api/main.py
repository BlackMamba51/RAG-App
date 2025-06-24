from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipelines.main import retriever, rag_pipeline
from api.schema import QuestionRequest, AnswerResponse
from api.web_ui import get_ui

app = FastAPI(
    title='RAG API',
    description='Answer the questions about art',
    version='1.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/ask', response_model=AnswerResponse, tags=['Query'])
def ask(data: QuestionRequest):
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым.")
    result = rag_pipeline(data.question, retriever)

    return {
        'answer': result['answer'],
        'context': result['context'],
        'sources': result['sources'],
        'trace': result['trace']
    }
app.get("/", include_in_schema=False)(get_ui)