from retriever import RetrieverEngine
from prompts import PromptTemplate
from llm_loader import LLMEngine
# -------------------- RAG Engine --------------------
class RAGEngine:
    def __init__(self, retriever, prompts, llm):
        self.retriever = retriever
        self.prompts = prompts
        self.llm = llm

    def answer(self, query):
        rewritten = self.llm.run(self.prompts.rewrite_prompt, {"question": query})
        docs = self.retriever.retrieve(rewritten)
        context = "\n\n".join([d.page_content for d in docs])
        return self.llm.run(
            self.prompts.answer_prompt,
            {"context": context, "question": rewritten}
        )

    def stream_answer(self, query):
        rewritten = self.llm.run(self.prompts.rewrite_prompt, {"question": query})
        docs = self.retriever.retrieve(rewritten)
        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = self.prompts.answer_prompt.format(context=context, question=rewritten)
        for chunk in self.llm.stream(final_prompt):
            yield chunk

