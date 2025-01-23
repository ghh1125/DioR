def fix_unclosed_quotes(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if line.count('"') % 2 != 0:
                line = line.rstrip('\n') + '"\n'
                print(f"Fixed unclosed quote at line {i+1}")
            outfile.write(line)

input_file = './data/dpr/psgs_w100.tsv'
output_file = './data/dpr/psgs_w100_fixed.tsv'

fix_unclosed_quotes(input_file, output_file)
