# Learn-AI-Practice

本仓库旨在记录 AI 学习相关的实践代码。项目使用 VSCode 编辑器和 Remote Development 插件，方便在 Docker 容器内进行开发。

环境配置见 [.devcontainer](/.devcontainer/devcontainer.json)，为解决容器重启与持久化数据的冲突，项目使用了几个特别的子路径（需要在项目根目录下创建）：
- `HF_HOME`: 存储 HuggingFace 的文件，避免重复拉取模型。
- `site-packages`: 存储 Python 的第三方依赖包，类似与 node.js 中的 node_modules 目录。

## 子主题

每个 AI 主题分隔为独立的子目录：
- [llm-chat](/llm-chat): 加载 LLM 并生成对话响应。
- [llm-fine-tunning-k8sdoc](/llm-fine-tunning-k8sdoc/): 以翻译 Kubernetes 文档为例，探索 LLM 的 Fine-Tuning 微调。
- [RAG-langchain-redis-llama.cpp](/RAG-langchain-redis-llama.cpp/): 使用 `langchain`、`redis`、`llama.cpp` 构建 RAG 示例。