import json
from typing import Dict

# ------------------- Existing imports -------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

# ------------------- 1. Load PDF -------------------
loader = PyPDFLoader("sa.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(documents)

# ------------------- 2. Embeddings + FAISS -------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(all_splits, embeddings)
vectorstore.save_local("faiss_index_")
db = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ------------------- 3. Prompts -------------------
rewrite_prompt = PromptTemplate(
    template="""Rewrite the user question into a better search query.

Original Question: {question}
""",
    input_variables=["question"]
)

answer_prompt = PromptTemplate(
    template="""
You are a STRICT RAG model.

RULES:
1. You MUST answer ONLY using the provided context.
2. If the answer is NOT in the context, say ONLY: "I don't know".
3. DO NOT use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
""",
    input_variables=["context", "question"]
)

# ------------------- 4. LLM -------------------
llm = OllamaLLM(model="deepseek-r1:1.5b")

# ------------------- 5. Helper: structured query -------------------
class RAGProcessor:
    def __init__(self, llm, retriever, answer_prompt, rewrite_prompt):
        self.llm = llm
        self.retriever = retriever
        self.answer_prompt = answer_prompt
        self.rewrite_prompt = rewrite_prompt

    def answer_query(self, structured_context: Dict, question: str) -> str:
        """
        Given a structured context (dict or Pydantic model), format and query LLM
        """
        # Ensure context is JSON string
        try:
            context_text = json.dumps(structured_context, indent=2)
        except Exception:
            context_text = str(structured_context)

        # Option 1: format manually using answer_prompt template
        final_prompt = self.answer_prompt.format(context=context_text, question=question)

        # Run LLM
        response = self.llm.invoke(final_prompt)
        return response

    def rag_retrieve_context(self, question: str) -> str:
        """Retrieve context from FAISS and combine as a string"""
        rewritten = (self.rewrite_prompt | self.llm).invoke({"question": question})
        # Use the vectorstore directly
        docs = self.retriever.invoke(rewritten)
        context = "\n\n".join([d.page_content for d in docs])
        return context, rewritten

# ------------------- 6. Interactive chat -------------------
if __name__ == "__main__":
    processor = RAGProcessor(llm, retriever, answer_prompt, rewrite_prompt)

    # Example: structured context from PDF (can be a dict or model)
    structured_context = {"sections": [d.page_content for d in all_splits]}

    print("\n💬 Entering interactive chat mode. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Take care! 👋")
            break

        # Optionally, you can use FAISS to get retrieved context instead of full structured context
        context_text, rewritten_question = processor.rag_retrieve_context(user_input)

        answer = processor.answer_query(context_text, user_input)
        print("\nChatbot:", answer, "\n")
