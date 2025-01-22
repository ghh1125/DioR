# 导入 pandas 库
import pandas as pd

# 文件路径
file_path = '/home/disk_16T/ghh/data/dpr/psgs_w100_fixed.tsv'

# 使用 pandas 读取文件，并计算行数
df = pd.read_csv(file_path, delimiter='\t')
num_records = len(df)

# 打印结果
print(f"文件 {file_path} 中共有 {num_records} 条记录。")
