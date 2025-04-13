import re
import csv

# 读取combined_log.txt文件
with open("combined_log.txt", "r") as file:
    text = file.read()

# 正则匹配所有测试用例
pattern = r"log_size=(\d+)\s+precomp=(\d+)\s+c=(\d+).*?msm time : ([\d.]+) ms"
matches = re.findall(pattern, text, re.DOTALL)

# 收集所有结果并按log_size分组
results = {}
for match in matches:
    log_size, precomp, c, msm_time = match
    log_size = int(log_size)
    msm_time = float(msm_time)
    if log_size not in results:
        results[log_size] = []
    results[log_size].append((int(precomp), int(c), msm_time))

# 找出每个log_size的最低msm_time
min_times = {}
for log_size, entries in results.items():
    min_times[log_size] = min(entries, key=lambda x: x[2])

# 写入 CSV 文件
with open("msm_results_min.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["log_size", "precomp", "c", "msm_time(ms)"])  # 表头
    
    # 按log_size排序，只输出每个log_size的最小值
    for log_size in sorted(min_times.keys()):
        precomp, c, msm_time = min_times[log_size]
        writer.writerow([log_size, precomp, c, msm_time])