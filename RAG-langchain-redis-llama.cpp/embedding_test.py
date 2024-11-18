import unittest
from embedding import Embedding  # 确保文件名为 embedding.py

class TestEmbedding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """初始化嵌入模型"""
        cls.embedder = Embedding()

    def test_embedding_cases(self):
        """使用多种输入场景测试嵌入生成"""
        cases = [
            # 测试单个句子
            {"input": "This is a test sentence.", "expected_length": 1},
            # 测试多个句子
            {"input": ["First sentence.", "Second sentence."], "expected_length": 2},
            # 测试单个句子以列表形式输入
            {"input": ["A single sentence in a list."], "expected_length": 1},
        ]

        for case in cases:
            with self.subTest(case=case):
                embeddings = self.embedder.text2vector(case["input"])
                self.assertEqual(len(embeddings), case["expected_length"], 
                                 f"Expected {case['expected_length']} embeddings, got {len(embeddings)}.")

    def test_invalid_inputs(self):
        """测试无效输入场景"""
        invalid_cases = [
            "",       # 空字符串
            [],       # 空列表
            None      # None 值
        ]

        for invalid_input in invalid_cases:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError):
                    self.embedder.text2vector(invalid_input)

    def test_invalid_model(self):
        """测试加载无效模型"""
        with self.assertRaises(ValueError):
            Embedding(model="invalid-model-name")

if __name__ == "__main__":
    unittest.main()
