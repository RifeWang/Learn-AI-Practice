import unittest
from vectordb import VectorDB


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        """
        初始化测试环境，连接到 Redis 并设置测试索引。
        """
        self.db = VectorDB()
        self.index_name = "test_index"
        self.vector_dim = 4

        # 创建测试索引
        self.db.create_index(self.index_name, self.vector_dim)

        # 测试数据
        self.sample_data = [
            {"id": "1", "text": "First sample text", "vector": [0.1, 0.2, 0.3, 0.4]},
            {"id": "2", "text": "Second sample text", "vector": [0.4, 0.3, 0.2, 0.1]},
            {"id": "3", "text": "Another sample text", "vector": [0.5, 0.5, 0.5, 0.5]},
        ]

        # 添加测试数据
        for item in self.sample_data:
            self.db.add(self.index_name, item["id"], item["text"], item["vector"])

    def tearDown(self):
        """
        清理测试环境，删除测试索引和数据。
        """
        self.db.redis_client.execute_command(f"FT.DROPINDEX {self.index_name} DD")

    def test_add_and_search(self):
        """
        测试添加数据项后，是否能够正确检索。
        """
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = self.db.search(self.index_name, query_vector, top_k=2)

        self.assertEqual(len(results), 2, "Should return 2 results")
        self.assertEqual(results[0]["id"], "1", "First result should match the most similar vector")
        self.assertTrue("_score" in results[0], "Result should contain similarity score")

    def test_delete(self):
        """
        测试删除数据项后，是否无法检索到该数据项。
        """
        # 删除数据项
        self.db.delete(self.index_name, "1")

        # 尝试检索被删除的数据
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = self.db.search(self.index_name, query_vector, top_k=3)

        self.assertTrue(all(result["id"] != "1" for result in results), "Deleted item should not appear in results")


if __name__ == "__main__":
    unittest.main()
