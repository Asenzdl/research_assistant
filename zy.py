from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from my_llm import ollama_qwen3_embeddings
from langchain_chroma import Chroma
import uuid

class KnowledgeBase:
    def __init__(self, persist_dir: str = "./db/chroma_db"):
        self.embeddings = ollama_qwen3_embeddings
        self.vectorstore = Chroma(
            collection_name="research_kb",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )   # 存小 chunk（child）
        self.docstore = {}  # 普通 dict 就够，存大 chunk（parent）
        self.all_small_chunks = []  # 给 BM25 用，存所有小 chunk

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=80
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )

    def _load_source(self, source: str) -> list:
        if source.startswith("http://") or source.startswith("https://"):
            return WebBaseLoader(source).load()
        elif source.endswith(".pdf"):
            return PyPDFLoader(source).load()
        else:
            raise ValueError(f"暂不支持的文件类型：{source}")



    def add_documents(self, sources: list[str]) -> int:
        # 加载 → 切分 → 入库
        # 返回成功入库的文档数量
        loader = []
        for source in sources:
            loader.extend(self._load_source(source))
        parent_docs = self.parent_splitter.split_documents(loader)

        add_counts = 0  # 统计父文档数量
        all_children = []
        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            parent_doc.metadata["parent_id"] = parent_id
            self.docstore[parent_id] = parent_doc
            add_counts += 1
            child_doc = self.child_splitter.split_documents([parent_doc])
            all_children.extend(child_doc)
        self.vectorstore.add_documents(all_children)
        self.all_small_chunks.extend(all_children)
        return add_counts   # 13




    def as_retriever(self):
        # 返回 EnsembleRetriever（向量 + BM25）
        # 内部用 ParentDocumentRetriever 作为向量检索的基础
        ...


if __name__ == '__main__':
    a = KnowledgeBase()
    sources = ["https://python.langchain.com/docs/introduction/",
               "./asset/db.pdf"]
    print(a.add_documents(sources))
