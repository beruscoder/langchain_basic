# -------------------- Text Splitter --------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

    def split(self, documents):
        return self.splitter.split_documents(documents)
    
class VectorStoreManager:
    def __init__(self, index_name="faiss_index_", embed_model="nomic-embed-text"):
        self.index_name = index_name
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vectorstore = None

    def build(self, splits):
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.vectorstore.save_local(self.index_name)
        return self.vectorstore

    def load(self):
        self.vectorstore = FAISS.load_local(
            self.index_name,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore
