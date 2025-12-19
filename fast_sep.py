from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import all classes from your class-based RAG pipeline
from pdf_loader import DocumentLoader
from textsplitter import TextSplitter,VectorStoreManager
from retriever import RetrieverEngine
from prompts import PromptManager
from llm_loader import LLMEngine
from rag import RAGEngine

app = FastAPI(title="RAG FastAPI Service")

# -------------------- BUILD FULL RAG PIPELINE --------------------

# 1. Load PDF
loader = DocumentLoader("sa.pdf")
docs = loader.load()

# 2. Split
splitter = TextSplitter()
splits = splitter.split(docs)

# 3. Vector Store
vs = VectorStoreManager()
vs.build(splits)
db = vs.load()

# 4. Retriever
retriever = RetrieverEngine(db)

# 5. Prompts
prompts = PromptManager()

# 6. LLM Engine
llm = LLMEngine()

# 7. RAG Engine
rag_engine = RAGEngine(retriever, prompts, llm)


# ------------------- REQUEST MODEL -------------------
class Query(BaseModel):
    question: str


# ------------------- NORMAL (NON-STREAMING) ENDPOINT -------------------
@app.post("/rag")
async def rag_endpoint(payload: Query):
    answer = rag_engine.answer(payload.question)
    return {"answer": answer}


# ------------------- STREAMING RAG ENDPOINT -------------------
@app.post("/rag_stream")
async def rag_stream_api(payload: Query):
    def event_stream():
        for chunk in rag_engine.stream_answer(payload.question):
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")


# ------------------- RUN SERVER -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast:app", host="0.0.0.0", port=8000, reload=True)
 