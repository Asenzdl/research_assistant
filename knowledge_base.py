from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # 初始化 vectorstore、docstore、splitters、embeddings
        ...

    def add_documents(self, sources: list[str]) -> int:
        # 加载 → 切分 → 入库
        # 返回成功入库的文档数量
        ...

    def as_retriever(self):
        # 返回 EnsembleRetriever（向量 + BM25）
        # 内部用 ParentDocumentRetriever 作为向量检索的基础
        ...


# 1. 加载文档
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()
# print(type(docs))   # List[Document]


splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    # separators=[
    #     "\n\n",    # 1. 双换行（段落分隔）
    #     "\n",      # 2. 单换行
    #     ". ",      # 3. 句号+空格（完整句子）
    #     "? ",      # 4. 问号+空格
    #     "! ",      # 5. 感叹号+空格
    #     "; ",      # 6. 分号+空格
    #     ": ",      # 7. 冒号+空格
    #     ", ",      # 8. 逗号+空格
    #     " ",       # 9. 空格
    #     ""         # 10. 字符级（最后手段）
    # ]
)

chunks = splitter.split_documents(docs)  # 输入 List[Document]，输出 List[Document]
# print(f"分块数：{len(chunks)}")
# print(chunks[0].page_content)
# print(chunks[0].metadata)

