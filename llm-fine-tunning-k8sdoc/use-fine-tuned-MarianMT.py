from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载微调后的模型
model_name = 'opus-mt-en-zh-finetuned'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# {"en": "The kubelet uses heuristics to retrieve logs. This helps if you are not aware whether a given system service is\nwriting logs to the operating system's native logger like journald or to a log file in `/var/log/`. The heuristics\nfirst checks the native logger and if that is not available attempts to retrieve the first logs from\n`/var/log/<servicename>` or `/var/log/<servicename>.log` or `/var/log/<servicename>/<servicename>.log`.\n\nThe complete list of options that can be used are:", "zh": "kubelet 使用启发方式来检索日志。\n如果你还未意识到给定的系统服务正将日志写入到操作系统的原生日志记录程序（例如 journald）\n或 `/var/log/` 中的日志文件，这会很有帮助。这种启发方式先检查原生的日志记录程序，\n如果不可用，则尝试从 `/var/log/<servicename>`、`/var/log/<servicename>.log`\n或 `/var/log/<servicename>/<servicename>.log` 中检索第一批日志。\n\n可用选项的完整列表如下："}
# {"en": "This document highlights and consolidates configuration best practices that are introduced\nthroughout the user guide, Getting Started documentation, and examples.", "zh": "本文档重点介绍并整合了整个用户指南、入门文档和示例中介绍的配置最佳实践。"}

# 使用模型生成中文翻译
input_en = "This document highlights and consolidates configuration best practices that are introduced\nthroughout the user guide, Getting Started documentation, and examples."
inputs = tokenizer(input_en, return_tensors='pt')
outputs = model.generate(**inputs, max_length=512)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)