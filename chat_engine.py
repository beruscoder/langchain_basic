class ChatEngine:
    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.history = []  # conversation memory

    def build_context(self):
        text = ""
        for turn in self.history:
            text += f"USER: {turn['user']}\nASSISTANT: {turn['assistant']}\n\n"
        return text.strip()

    # ------------ Non-Streaming Chat ------------
    def ask(self, query: str):
        history_text = self.build_context()

        # Pass history to RAG engine
        answer = self.rag.answer(query, history_text)

        # Store to memory
        self.history.append({"user": query, "assistant": answer})

        return answer

    # ------------ Streaming Chat ------------
    def ask_stream(self, query: str):
        history_text = self.build_context()

        buffer = ""  # Collect full answer

        # Stream from RAG engine
        for chunk in self.rag.stream_answer(query, history_text):
            buffer += chunk
            yield chunk

        # After streaming, store final output
        self.history.append({"user": query, "assistant": buffer})
