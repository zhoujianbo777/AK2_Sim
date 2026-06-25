import os, re
roots = [r'F:\CODE\AK2_Sim\views', r'F:\CODE\AK2_Sim\modules', r'F:\CODE\AK2_Sim\tools']
files = [r'F:\CODE\AK2_Sim\sim_main.py']
for root in roots:
    for f in os.listdir(root):
        if f.endswith('.py'):
            files.append(os.path.join(root, f))
pat = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+')
hits = 0
for fp in files:
    lines = open(fp, encoding='utf-8').readlines()
    for i, l in enumerate(lines, 1):
        if pat.search(l):
            # Only print file/line; avoid printing Chinese to GBK terminal
            print(f'{fp}:{i}: [CHINESE FOUND]')
            hits += 1
print(f'Total hits: {hits}')
