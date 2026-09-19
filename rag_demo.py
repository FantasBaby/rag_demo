from typing import List
from sentence_transformers import SentenceTransformer

# 拆分成片段
def split_into_chunks(doc_file):
    with open(doc_file,'r',encoding='utf-8') as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]

chunks = split_into_chunks("doc.md")

for i ,chunk in enumerate(chunks):
    print(f"[{i}] {chunk}\n")

#     ------------------------------------------------拆分片段完毕

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
# 参数chunk就是上面拆解的片段  返回对应的向量
def embed_chunk(chunk):
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()

#     ------------------------------------------------检查一下返回的向量
test_embedding = embed_chunk("测试内容")
print(len(test_embedding))
print(test_embedding)
#     ------------------------------------------------遍历  将所有的片段都变为向量
embeddings = [embed_chunk(chunk) for chunk in chunks]
#     ------------------------------------------------检查一下返回的向量
print(len(embeddings))
print(embeddings[0])

#     ------------------------------------------------检查一下返回的向量
import chromadb
#     ------------------------------------------------数据存放在  chroma.db
chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection(name="default")
#     ------------------------------------------------所以的片段和向量存放
def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )

save_embeddings(chunks, embeddings)

#     ------------------------------------------------召回函数  参数query是用户的问题  top_k是召回的记录数量

def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

query = "哆啦A梦使用的3个秘密道具分别是什么？"
retrieved_chunks = retrieve(query, 5)
#     ------------------------------------------------查看返回的最为相关的 5 个片段
for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i}] {chunk}\n")


#     ------------------------------------------------重新排序 按照得分排名
from sentence_transformers import CrossEncoder

def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]
#     ------------------------------------------------前 5 个排名 取前 3
reranked_chunks = rerank(query, retrieved_chunks, 3)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")

#     ------------------------------------------------调用api  根据提示词  输出最终结果

from dotenv import load_dotenv
from google import genai

load_dotenv()
google_client = genai.Client()

def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
    用户问题: {query}
    相关片段:
    {"\n\n".join(chunks)}
    请基于上述内容作答，不要编造信息。"""
    print(f"{prompt}\n\n---\n")
    response = google_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

answer = generate(query, reranked_chunks)
print(answer)
