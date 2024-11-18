from sentence_transformers import SentenceTransformer

class Embedding:
    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    def __init__(self, model="sentence-transformers/all-mpnet-base-v2"):
        """
        初始化嵌入模型。

        :param model: 模型名称或路径，默认为 "sentence-transformers/all-mpnet-base-v2"。
        """
        self.model_name = model
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model '{model}'. Ensure the model exists and is valid. Error: {e}")

    def text2vector(self, sentences):
        """
        获取输入句子的向量。

        :param sentences: 单个句子或句子列表。
        :return vectors: 向量列表。
        """
        if not sentences:
            raise ValueError("Input sentences cannot be empty.")
        
        if isinstance(sentences, str):
            sentences = [sentences]  # 转换为列表以统一处理

        try:
            vectors = self._model.encode(sentences)
            return vectors
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings. Error: {e}")
