def fix_unclosed_quotes(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if line.count('"') % 2 != 0:
                # 修复未关闭的引号，假设是添加缺失的引号
                line = line.rstrip('\n') + '"\n'
                print(f"Fixed unclosed quote at line {i+1}")
            outfile.write(line)

# 输入文件路径和输出文件路径
input_file = '/home/disk_16T/ghh/data/dpr/psgs_w100.tsv'
output_file = '/home/disk_16T/ghh/data/dpr/psgs_w100_fixed.tsv'

# 修复未关闭的引号
fix_unclosed_quotes(input_file, output_file)
