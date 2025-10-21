import random
import matplotlib.pyplot as plt
import csv

X = []
Y = []
labels = []

# 使用集合加速去重（比遍历列表更高效）
seen = set()

for x in range(-20,21):
    for y in range(-20,21):
        if x**2 + y**2 <= 400:  # 在半径为20的圆内（含边界）
            res=0
            if x**2+y**2<144:
                res=1
            X.append(x)
            Y.append(y)
            labels.append(res)

# 保存到 CSV 文件
with open('points.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'label'])  # 表头
    for x, y, label in zip(X, Y, labels):
        writer.writerow([x, y, label])

# 绘制散点图
colors = ['red' if label == 1 else 'blue' for label in labels]
plt.figure(figsize=(8, 8))
plt.scatter(X, Y, c=colors, s=20, alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-21, 21)
plt.ylim(-21, 21)
plt.title('Scatter Plot: Points inside Circle (r=20)\nRed: inside r=10, Blue: between r=10 and r=20')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"已生成 {len(X)} 个不重复点，并保存到 points.csv")