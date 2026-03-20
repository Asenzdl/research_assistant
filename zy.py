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
            docs = WebBaseLoader(source).load()
        else:
            if source.endswith(".pdf"):
                docs = PyPDFLoader(source).load()
        return docs



    def add_documents(self, sources: list[str]) -> int:
        # 加载 → 切分 → 入库
        # 返回成功入库的文档数量
        loader = []
        for source in sources:
            loader.extend(self._load_source(source))
        parent_docs = self.parent_splitter.split_documents(loader)

        add_counts = 0
        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            self.docstore[parent_id] = parent_doc
            add_counts += 1
            parent_doc.metadata["parent_id"] = parent_id
            child_doc = self.child_splitter.split_documents([parent_doc])
            self.vectorstore.add_documents(child_doc)
            self.all_small_chunks.extend(child_doc)
        return add_counts   # 13




    def as_retriever(self):
        # 返回 EnsembleRetriever（向量 + BM25）
        # 内部用 ParentDocumentRetriever 作为向量检索的基础
        ...


# loader1 = WebBaseLoader("https://python.langchain.com/docs/introduction/")
# docs1 = loader1.load()



# splitter1 = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=80,
# )

# chunks1 = splitter1.split_documents(docs1)
# print(len(chunks1)) # 16
# print(chunks1[0].metadata)  # {'source': 'https://python.langchain.com/docs/introduction/', 'title': 'LangChain overview - Docs by LangChain', 'description': 'LangChain is an open source framework with a prebuilt agent architecture and integrations for any model or tool—so you can build agents that adapt as fast as the ecosystem evolves', 'language': 'en'}
# print(chunks1[0].page_content)


# loader2 = PyPDFLoader("./asset/db.pdf")
# docs2 = loader2.load()
#
# splitter2 = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=80
# )
# chunks2 = splitter2.split_documents(docs2)
# print(len(chunks2))     # 24
# print(chunks2[0].metadata)  # {'producer': 'Typora', 'creator': 'Typora', 'creationdate': '20260319144557', 'moddate': '20260319144557', 'source': './asset/db.pdf', 'total_pages': 10, 'page': 0, 'page_label': '1'}
# print(chunks2[0].page_content)



if __name__ == '__main__':
    a = KnowledgeBase()
    sources = ["https://python.langchain.com/docs/introduction/",
               "./asset/db.pdf"]
    print(a.add_documents(sources))
