import pickle
import numpy as np


# 指定 pickle 文件路径
pickle_file_path = "/home/disk2/ghh/llm-hallucinations/results/open_llama_7b_place_of_birth_start-0_end-2500_12_18.pickle"


with open(pickle_file_path, "rb") as infile:
    # 从文件中读取并反序列化 pickle 数据
    results = pickle.loads(infile.read())

# 检查反序列化后的结果类型
print(type(results))  # 确认 `results` 是字典类型
print(results.keys())  # 查看字典中的键

# 从字典中提取 'correct' 键的值，并转换为 NumPy 数组
correct = np.array(results['correct'])
# 输出 'correct' 数组的前几个元素
print(correct[:10])  # 假设是一个数组，打印前 10 个元素
attributes_first = np.array(results['attributes_first'], dtype=object)

print(attributes_first)
