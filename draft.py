import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 数据
    categories = ['Go', 'Tennis', 'Star_craft', 'Badminton']
    values = [2.5409395216257638, 2.835894992068166, 2.951381802582708, 4.760913425273023]
    colors = ['blue', 'green', 'red', 'purple']  # 四种颜色

    # 绘制直方图
    plt.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)

    # 在每个柱子上显示数值
    for i, value in enumerate(values):
        plt.text(i, value + 0.1, f'{value:.2f}', ha='center', va='bottom')

    # 设置标题和轴标签
    plt.title("D value when M' = 0.5")
    plt.xlabel('Match Categories')
    plt.ylabel('D value')

    # 设置纵坐标范围
    plt.ylim(1, 6)

    # 显示图表
    plt.show()