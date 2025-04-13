import matplotlib.pyplot as plt
import numpy as np
import matplotlib


plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'  # 替换为你选择的字体# 解决负号显示问题
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体



# 准备数据（已修复原始数据中的数组长度不一致问题）
data = {
    "2²⁰": {
        "C": np.array([15,16,17,18,19,20]),
        "time": np.array([53.271,41.357,42.025,39.86,38.101,39.464])
    },
    "2²¹": {
        "C": np.array([16,17,18,19,20,21]),
        "time": np.array([78.421,74.967,73.834,70.43,75.01,91.43])
    },
    "2²²": {
        "C": np.array([16,17,18,19,20,21,22]),
        "time": np.array([152.849,140.839,145.506,138.264,133.747,147.815,177.808])
    }
}

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

# 子图(a) 2²⁰和2²¹
for scale, color, marker in zip(["2²⁰", "2²¹"], ['#1f77b4', '#ff7f0e'], ['o', 's']):
    ax1.plot(data[scale]["C"], data[scale]["time"], 
            marker=marker, color=color, linewidth=2, markersize=6,
            label=f'MSM {scale}')
    
ax1.set_title('(a) MSM尺度: 2²⁰和2²¹', fontsize=10)
ax1.set_xlabel('组大小 (BN254)', fontsize=9)
ax1.set_ylabel('运行时间 (ms)', fontsize=9)
ax1.set_ylim(35, 95)
ax1.set_xlim(14.5, 21.5)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=8)

# 子图(b) 2²²
for scale, color, marker in zip(["2²²"], ['#2ca02c'], ['^']):
    ax2.plot(data[scale]["C"], data[scale]["time"],
            marker=marker, color=color, linewidth=2, markersize=6,
            label=f'MSM {scale}')
    
ax2.set_title('(b) MSM尺度: 2²²', fontsize=10)  # 原始图片包含2²³，但数据中缺少
ax2.set_xlabel('组大小 (BN254)', fontsize=9)
ax2.set_ylabel('运行时间 (ms)', fontsize=9)
ax2.set_ylim(130, 180)
ax2.set_xlim(15.5, 22.5)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=8)

# 调整布局并显示
plt.tight_layout()
plt.show()