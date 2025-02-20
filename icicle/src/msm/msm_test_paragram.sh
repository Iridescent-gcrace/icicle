#!/bin/bash

# 配置参数
msm_log_sizes=(16 18 20 22)
precomp_factors=(10 12 14 16 18 20 22)  # 间隔2的等差数列
user_cs=(15 17 19 21 23 25)             # 间隔2的等差数列

# 创建输出目录
mkdir -p ./work/reports
mkdir -p ./work/logs

# 统一日志文件路径
LOG_FILE="./work/logs/combined_log.txt"

# 清空旧日志（如需保留历史日志可注释此句）
> "$LOG_FILE"

# 主测试循环
total=$(( ${#msm_log_sizes[@]} * ${#precomp_factors[@]} * ${#user_cs[@]} ))
current=0

for log_size in "${msm_log_sizes[@]}"; do
  for precomp in "${precomp_factors[@]}"; do
    for c in "${user_cs[@]}"; do
      ((current++))
      
      # 生成报告文件名
      report_name="msm_${log_size}_${precomp}_${c}"
      
      # 记录进度信息
      echo -e "\n\n[${current}/${total}] Testing: log_size=${log_size} precomp=${precomp} c=${c} @ $(date)"
      echo -e "====================== TEST START ======================"
      
      # 运行测试并追加日志（同时输出到终端和文件）
      {
        echo -e "\n\n[${current}/${total}] Testing: log_size=${log_size} precomp=${precomp} c=${c} @ $(date)"
        echo -e "====================== TEST START ======================"
        nsys profile --output "./work/reports/${report_name}" \
          ./work/test_msm $log_size 1 $precomp $c
        echo -e "====================== TEST END ======================\n"
      } | tee -a "$LOG_FILE"
    done
  done
done

echo -e "\nAll tests completed! Reports saved in ./work/reports/"
echo "Combined log file: $LOG_FILE"