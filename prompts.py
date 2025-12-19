# -------------------- Prompts --------------------
from langchain_core.prompts import PromptTemplate

class PromptManager:
    def __init__(self):
        self.rewrite_prompt = PromptTemplate(
            template="""Rewrite the user question into a better search query.

Original Question: {question}
""",
            input_variables=["question"]
        )

        self.answer_prompt = PromptTemplate(
            template="""
You are a STRICT RAG model.

RULES:
1. Answer ONLY using the provided context.
2. If answer is NOT in context → respond: "I don't know".
3. Do NOT use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
""",
            input_variables=["context", "question"]
        )
