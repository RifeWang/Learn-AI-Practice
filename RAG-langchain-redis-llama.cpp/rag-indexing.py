# 1. 加载文档，此处是抓取网页的特定标签内容
import bs4
from langchain_community.document_loaders import WebBaseLoader

bs4_strainer = bs4.SoupStrainer(class_=("td-content"))
loader = WebBaseLoader(
    web_paths=("https://kubernetes.io/docs/concepts/services-networking/service/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

print("Doc length:", len(docs))
print("Page-content length:", len(docs[0].page_content))


# 2. 切割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print("Chunk size:", len(all_splits))
print("First chunk:", all_splits[0])


# 3. 向量化
from embedding import Embedding

embed = Embedding()
dataset = []
for i in range(len(all_splits)):
    chunk = all_splits[i]
    dataset.append({
        "id": i,
        "text": chunk.page_content,
        "vector": embed.text2vector(chunk.page_content)[0].tolist() # 将 ndarray 转换为 list
    })

print("Dataset first element:", dataset[0])


# 4. 存储至向量数据库
from vectordb import VectorDB

index_name = "rag_test"
vector_dim = len(dataset[0]["vector"])

vectordb = VectorDB()
vectordb.create_index(index_name, vector_dim)

# 将数据存入数据库
for item in dataset:
    vectordb.add(index_name, item["id"], item["text"], item["vector"])

print(f"Data successfully added to the vector database with index: {index_name}")