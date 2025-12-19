# -------------------- Retriever --------------------
class RetrieverEngine:
    def __init__(self, vectorstore, k=5):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve(self, query):
        return self.retriever.invoke(query)
