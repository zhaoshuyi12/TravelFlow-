import re
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

faq_text=None
with open('../order_faq.md', encoding="utf-8")as f:
    faq_text=f.read()

# 2. 按 "## 标题" 切分（保留分隔符）
chunks = re.split(r"(?=\n## )", faq_text)
# 移除空 chunk
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# 3. 转为 LangChain Document 对象（带 metadata）
docs = [
    Document(page_content=chunk, metadata={"source": "order_faq.md", "chunk_id": i})
    for i, chunk in enumerate(chunks)
]
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
#创建文件的向量集
vectorstore = FAISS.from_documents(docs, embeddings_model)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # 默认返回 top-5
)

@tool
def lookup_policy(query: str) -> str:
    """查询公司政策，检查某些选项是否允许。
    在进行航班变更或其他'写'操作之前使用此函数。"""
    # 查询相似度最高的 k 个文档
    docs = retriever.invoke(query)
    # 返回这些文档的内容
    return "\n\n".join([doc["page_content"] for doc in docs])

if __name__ == '__main__':
    print(lookup_policy)