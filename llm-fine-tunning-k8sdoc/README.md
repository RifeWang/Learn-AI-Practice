# 以翻译 Kubernetes 文档为例，探索 LLM 的 Fine-Tuning 微调

1. 准备数据集：使用 `prepare-dataset.py` 生成数据集，例如 `docs_concepts_dataset.jsonl`
2. 使用特定领域模型（基于 MarianMT 框架的翻译模型）做微调：`fine-tuning-MarianMT.py`
3. 使用通用的 LLM 做微调：`fine-tuning-Qwen2.5-0.5B.py`
4. 可以使用 loRA 加速