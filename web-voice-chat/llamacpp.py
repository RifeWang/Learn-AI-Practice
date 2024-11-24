import requests
import json


class LlamaCppClient:
    def __init__(self, host="http://host.docker.internal", port=8080):
        """
        初始化 Llama.cpp 客户端。
        
        :param host: 服务的主机地址，默认是 "http://localhost"。
        :param port: 服务的端口号，默认是 8080。
        """
        self.base_url = f"{host}:{port}"

    def completion(self, prompt):
        """
        调用 HTTP API 获取嵌入向量。

        :param prompt: 输入文本。
        :return: 响应。
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """
                        You are a friendly conversation partner. Be natural, engaging, and helpful in our discussions. Respond to questions clearly and follow the conversation flow naturally.
                    """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()  # 返回 JSON 响应
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

