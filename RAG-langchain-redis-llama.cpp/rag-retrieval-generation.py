# 1. Retrieval
question = "What is service?"

from embedding import Embedding
embed = Embedding()
question_vec = embed.text2vector(question)[0].tolist() # 将 ndarray 转换为 list

# 检索相关文档
from vectordb import VectorDB
index_name = "rag_test"
vectordb = VectorDB()
retrieved_docs = vectordb.search(index_name, question_vec, 6)
# print(retrieved_docs)

# 生成 prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

Context: {context}

Question: {question}

Helpful Answer:"""

prompt_without_context = template.format(context="", question=question)

context = "\r\n".join([doc["text"] for doc in retrieved_docs]) # 拼接 context
prompt = template.format(context=context, question=question) # 填充模板
print("Prompt: ", prompt)

# 2. Generation
from llamacpp import LlamaCppClient
client = LlamaCppClient()

responce_without_rag = client.completion(prompt_without_context)
answer_without_rag = responce_without_rag["choices"][0]["message"]["content"]

responce = client.completion(prompt)
answer = responce["choices"][0]["message"]["content"]

print("--------------------")
print("LLM without RAG Answer: \n", answer_without_rag)
print("--------------------")
print("LLM with RAG Answer: \n", answer)