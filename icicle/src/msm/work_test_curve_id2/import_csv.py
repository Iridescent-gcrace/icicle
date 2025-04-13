import re
import csv

# 读取combined_log.txt文件
with open("msm_combined_log.txt", "r") as file:
    data = file.read()


import re
import pandas as pd

# 原始数据

# 正则表达式模式
pattern = re.compile(
    r"running msm.*?2\^(\d+).*?precomp_factor=(\d+).*?c=(\d+).*?msm time : (\d+\.\d+) ms",
    re.DOTALL
)

# 数据提取
results = []
for match in pattern.finditer(data):
    exponent = int(match.group(1))
    precomp = int(match.group(2))
    c_value = int(match.group(3))
    time = float(match.group(4))
    
    results.append({
        "Exponent (2^n)": exponent,
        "Precomp Factor": precomp,
        "C": c_value,
        "MSM Time (ms)": time
    })

# 生成表格
df = pd.DataFrame(results)
with open("msm_results_min.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["log_size", "precomp", "c", "msm_time(ms)"])  # 表头
    writer.writerows(df.values)




# print(df.to_markdown(index=False))

# 正则匹配所有测试用例
# pattern = r"log_size=(\d+)\s+precomp=(\d+)\s+c=(\d+).*?msm time : ([\d.]+) ms"
# matches = re.findall(pattern, text, re.DOTALL)

# # 收集所有结果并按log_size分组
# results = {}
# for match in matches:
#     log_size, precomp, c, msm_time = match
#     log_size = int(log_size)
#     msm_time = float(msm_time)
#     if log_size not in results:
#         results[log_size] = []
#     results[log_size].append((int(precomp), int(c), msm_time))

# # 找出每个log_size的最低msm_time
# min_times = {}
# for log_size, entries in results.items():
#     min_times[log_size] = min(entries, key=lambda x: x[2])

# # 写入 CSV 文件
# with open("msm_results_min.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["log_size", "precomp", "c", "msm_time(ms)"])  # 表头
    
#     # 按log_size排序，只输出每个log_size的最小值
#     for log_size in sorted(min_times.keys()):
#         precomp, c, msm_time = min_times[log_size]
#         writer.writerow([log_size, precomp, c, msm_time])