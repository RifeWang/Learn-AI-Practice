import redis
from redis.commands.search.query import Query
from typing import List, Dict, Any


class VectorDB:
    """
    向量数据库封装类，底层使用 RedisJSON 和 Redisearch 实现。
    每个数据项包括 id、text、vector 三个元素。
    """
    # 由于使用了 VSCode devcontainer 容器环境进行开发，此处需要访问宿主机 Redis，所以 host 是特殊值
    def __init__(self, redis_host="host.docker.internal", redis_port=6379, redis_password=None):
        """
        初始化 Redis 连接。
        :param redis_host: Redis 主机地址
        :param redis_port: Redis 端口号
        :param redis_password: Redis 密码
        """
        self.redis_client = redis.StrictRedis(
            host=redis_host, port=redis_port, password=redis_password, decode_responses=True
        )

    def create_index(self, redis_index_name: str, vector_dim: int):
        """
        创建 Redis 向量索引。
        :param redis_index_name: 索引名称
        :param vector_dim: 向量的维度
        """
        try:
            # 删除已有索引（如果存在）
            self.redis_client.execute_command(f"FT.DROPINDEX {redis_index_name} DD")
        except redis.exceptions.ResponseError:
            pass  # 索引不存在时忽略错误

        # 创建索引
        self.redis_client.execute_command(
            f"FT.CREATE {redis_index_name} "
            f"ON JSON "
            f"PREFIX 1 {redis_index_name}: "
            f"SCHEMA "
            f"$.vector AS vector VECTOR FLAT 6 TYPE FLOAT32 DIM {vector_dim} DISTANCE_METRIC COSINE "
            f"$.text AS text TEXT"
        )

    def add(self, redis_index_name: str, item_id: str, text: str, vector: List[float]):
        """
        向 Redis 添加一个数据项。
        :param redis_index_name: 索引名称
        :param item_id: 数据项的唯一标识符
        :param text: 数据项的文本信息
        :param vector: 向量表示（列表）
        """
        try:
            # 构造 JSON 数据并写入 Redis
            key = f"{redis_index_name}:{item_id}"
            self.redis_client.json().set(key, "$", {"text": text, "vector": vector})
        except Exception as e:
            raise RuntimeError(f"Failed to add data to Redis. Error: {e}")

    def search(self, redis_index_name: str, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        根据查询向量检索最相似的数据项。
        :param redis_index_name: 索引名称
        :param query_vector: 查询向量（列表）
        :param top_k: 返回的最相似结果数
        :return: 包含 id、text 和相似度的字典列表
        """
        try:
            query = (
                Query("*=>[KNN $top_k @vector $blob AS score]")
                .return_fields("text", "score")
                .paging(0, top_k)
                .sort_by("score")
                .dialect(2)
            )
            params = {
                "top_k": top_k,
                "blob": self._vector_to_bytes(query_vector),
            }
            # 执行搜索命令
            result = self.redis_client.ft(redis_index_name).search(query, query_params=params)
            # print(result)

            # 解析并返回结果
            search_results = []
            for doc in result.docs:
                doc_dict = {
                    "id": doc.id.split(":")[-1],    # 移除索引前缀
                    "text": doc.text,          # 从 JSON 中获取文本
                    "_score": doc.score   # 转换余弦距离为相似度
                }
                search_results.append(doc_dict)
            
            return search_results
        except Exception as e:
            raise RuntimeError(f"Failed to search vectors in Redis. Error: {e}")

    def delete(self, redis_index_name: str, item_id: str):
        """
        删除 Redis 中的指定数据项。
        :param redis_index_name: 索引名称
        :param item_id: 数据项的唯一标识符
        """
        try:
            key = f"{redis_index_name}:{item_id}"
            self.redis_client.delete(key)
        except Exception as e:
            raise RuntimeError(f"Failed to delete data from Redis. Error: {e}")

    @staticmethod
    def _vector_to_bytes(vector: List[float]) -> bytes:
        """
        将向量转换为 Redis 可存储的字节形式。
        :param vector: 向量（列表）
        :return: 字节形式的向量
        """
        import struct
        return struct.pack(f"{len(vector)}f", *vector)
