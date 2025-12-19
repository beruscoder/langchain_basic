from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from new import rag, rag_stream   # <-- using the functions we created

app = FastAPI(title="RAG FastAPI Service")


# ------------------- NORMAL RAG ENDPOINT -------------------
class Query(BaseModel):
    question: str


@app.post("/rag")
async def rag_endpoint(data: Query):
    answer = rag(data.question)
    return {"answer": answer}


# ------------------- STREAMING RAG ENDPOINT -------------------
@app.post("/rag_stream")
async def rag_stream_api(payload: Query):
    def event_stream():
        for chunk in rag_stream(payload.question):
            print(chunk)
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")

# ------------------- RUN SERVER -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast:app", host="0.0.0.0", port=8000, reload=True)
