import os
import json
import re

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text(content):
    # 匹配模式：<!-- 英文 -->中文
    pattern = r'<!--(.*?)-->\s*(.*?)(?=<!--|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    pairs = []
    for match in matches:
        english = match.group(1).strip()
        chinese = match.group(2).strip()
        if english and chinese:
            pairs.append({
                'en': english,
                'zh': chinese
            })
    return pairs

def process_directory(zh_dir, output_file):
    """递归处理目录及其子目录"""
    parallel_data = []

    for root, dirs, files in os.walk(zh_dir):
        for file in files:
            if file.endswith('.md'):
                zh_file_path = os.path.join(root, file)
                zh_file_content = read_file_content(zh_file_path)
                en_zh_arr = extract_text(zh_file_content)
                parallel_data.extend(en_zh_arr)
                
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in parallel_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已生成并保存: {output_file}")

def process_multi_subdirs(zh_dir_root, zh_subdirs):
    """处理多个子目录，生成多个 JSON 文件"""
    for subdir in zh_subdirs:
        zh_dir = os.path.join(zh_dir_root, subdir)
        output_file = f"{subdir.replace('/', '_')}_dataset.jsonl"

        if not os.path.exists(zh_dir):
            print(f"中文目录不存在，跳过: {zh_dir}")
            continue

        process_directory(zh_dir, output_file)


# 需要下载 https://github.com/kubernetes/website 文档仓库至本地
# 指定中文根目录、子目录
zh_dir_root = '~/Desktop/github/k8s-website/content/zh-cn/'
zh_subdirs = ['docs/concepts']

process_multi_subdirs(zh_dir_root, zh_subdirs)