# -------------------- LLM Loader --------------------
from langchain_ollama import OllamaLLM
import os


class LLMEngine:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.llm = OllamaLLM(model=model_name)
        print(f"Initialized Ollama LLM with model: {model_name}")

    def run(self, prompt_obj, data):
        return (prompt_obj | self.llm).invoke(data)

    def stream(self, final_prompt):
        for chunk in self.llm.stream(final_prompt):
            yield str(chunk)
