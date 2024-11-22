
参考官方文档：https://github.com/ggerganov/whisper.cpp

1. 拉取镜像:
- `docker pull ghcr.io/ggerganov/whisper.cpp:main`
- `docker pull ghcr.io/ggerganov/whisper.cpp:main-cuda`: 支持 CUDA

2. 下载语音模型：
- 运行 docker 中的脚本下载：
```sh
# 注意挂载本地目录以持久化存储模型
docker run -it --rm \
  -v ~/ai-models/whisper:/models \
  ghcr.io/ggerganov/whisper.cpp:main \
  "./models/download-ggml-model.sh base /models"
```
- 或者直接去 https://huggingface.co/ggerganov/whisper.cpp 手动下载

3. 运行：
```sh
docker run -it --rm \
  -v ~/ai-models/whisper:/models \
  ghcr.io/ggerganov/whisper.cpp:main \
  "./main -m models/ggml-medium.en.bin -f samples/gb1.wav -t 8"
```