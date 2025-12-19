
from pdf_loader import DocumentLoader
from textsplitter import TextSplitter,VectorStoreManager
from retriever import RetrieverEngine
from prompts import PromptManager
from llm_loader import LLMEngine
from rag import RAGEngine

# -------------------- Main Runner --------------------
if __name__ == "__main__":
    # 1. Load PDF
    loader = DocumentLoader("sa.pdf")
    docs = loader.load()

    # 2. Split
    splitter = TextSplitter()
    splits = splitter.split(docs)

    # 3. VectorStore
    vs = VectorStoreManager()
    vs.build(splits)
    db = vs.load()

    # 4. Retriever
    retriever = RetrieverEngine(db)

    # 5. Prompts
    prompts = PromptManager()

    # 6. LLM
    llm = LLMEngine()

    # 7. RAG Engine
    rag = RAGEngine(retriever, prompts, llm)

    query = input("Ask: ")

    print("\n--- STREAMING OUTPUT ---\n")
    for c in rag.stream_answer(query):
        print(c, end="", flush=True)

    print("\n\n--- FINAL ANSWER ---\n")
    print(rag.answer(query))
