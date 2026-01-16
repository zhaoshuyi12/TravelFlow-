import re
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

faq_text=None
with open('../order_faq.md', encoding="utf-8")as f:
    faq_text=f.read()

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
#创建文件的向量集
class VectorStoreRetriever:
    def __init__(self,vector:list,docs:list):
        self.arr = np.array(vector)
        self.docs = docs
    @classmethod #类静态方法
    def from_docs(cls,docs:list):
        #从文档中生成嵌入向量
        vector_emd = embeddings_model.embed_documents([doc['page_content'] for doc in docs])
        vector=vector_emd
        return cls(vector,docs)

    def query(self,query:str,k:int=5):
        query_vector=embeddings_model.embed_query(query)
        query_arr = np.array(query_vector)
        scores=query_arr@self.arr.T #计算相似度
        top_k_idx = np.argpartition(scores, -k)[-k:]#排序 对相似度进行排序
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]#按分数降序排列的top-k索引

        return [{**self.docs[top_k],'similarity':scores[top_k]}for top_k in top_k_idx_sorted]

        print(scores)
retriever=VectorStoreRetriever.from_docs(docs)
@tool
def lookup_policy(query: str) -> str:
    """查询公司政策，检查某些选项是否允许。
    在进行航班变更或其他'写'操作之前使用此函数。"""
    # 查询相似度最高的 k 个文档
    docs = retriever.query(query, k=2)
    # 返回这些文档的内容
    return "\n\n".join([doc["page_content"] for doc in docs])

if __name__ == '__main__':
    print(lookup_policy)