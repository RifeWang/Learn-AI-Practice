from fastapi import FastAPI, WebSocket
import uvicorn
import whisper
import tempfile
import os
import signal

app = FastAPI()

# 系统环境需要安装 ffmpeg 命令行工具
# 加载 Whisper 模型，默认存储位置 ~/.cache/whisper，可以通过 download_root 设置
model = whisper.load_model("base", download_root="WHISPER_MODEL")

def LLMResponse(text):
    from llamacpp import LlamaCppClient
    client = LlamaCppClient()
    response = client.completion(text)
    response_text = response["choices"][0]["message"]["content"]
    return response_text

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        while True:
            # 接收音频数据
            audio_data = await websocket.receive_bytes()

            # 保存临时音频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # Whisper 语音识别
            result = model.transcribe(temp_audio_path)
            os.remove(temp_audio_path)  # 删除临时文件
            text = result["text"]
            print("user input: ", text)

            # 生成 AI 回复
            response_text = LLMResponse(text)
            print("AI response: ", response_text)

            # 发送回复
            await websocket.send_json({"input": text, "response": response_text})
    except Exception as e:
        print("Error: ", e)


def handle_shutdown(signal_num, frame):
    print(f"Received shutdown signal: {signal_num}")

def setup_signal_handlers():
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == "__main__":
    setup_signal_handlers()

    config = uvicorn.Config("main:app", port=8765, log_level="info")
    server = uvicorn.Server(config)
    server.run()