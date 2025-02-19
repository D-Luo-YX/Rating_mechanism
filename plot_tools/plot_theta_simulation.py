import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_difference_matrices(difference_matrices, matches, distributions):
    """
    绘制 4×3 个差异矩阵的上三角热度图，确保子图为正方形，且 colorbar 独立显示。

    参数：
    - difference_matrices: dict，键为 (match, distribution_type)，值为对应的差异矩阵 D_M。
    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))  # 4 行 3 列子图
    fig.suptitle("Difference Matrices Heatmaps (M - M')", fontsize=18)

    # 统一色阶范围
    all_values = np.concatenate([matrix.flatten() for matrix in difference_matrices.values()])
    vmin, vmax = np.min(all_values), np.max(all_values)

    for i, match in enumerate(matches):
        for j, dist in enumerate(distributions):
            ax = axes[i, j]
            matrix = difference_matrices.get((match, dist))

            # 创建上三角掩码（True 的部分会被遮盖）
            mask = np.tril(np.ones_like(matrix, dtype=bool))  # 下三角及对角线遮盖

            # 绘制上三角热度图
            im = sns.heatmap(matrix, ax=ax, mask=mask, cmap='coolwarm', vmin=vmin, vmax=vmax,
                             annot=False, cbar=False, square=True, linewidths=0.5, linecolor='white')

            ax.set_title(f"{match} - {dist}", fontsize=12)
            ax.set_xlabel("Player")
            ax.set_ylabel("Player")
            ax.set_aspect('equal')  # 确保正方形

    # 右侧统一 colorbar，调整位置防止重叠
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # 右侧稍微外移，避免挤压子图
    fig.colorbar(im.get_children()[0], cax=cbar_ax)

    # ✅ 替换 tight_layout()，手动调整布局
    plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)

    plt.show()

# def plot_difference_matrices(difference_matrices):
#     """
#     绘制 4×3 个差异矩阵的热度图，确保每个子图为正方形，colorbar 独立显示。
#
#     参数：
#     - difference_matrices: dict，键为 (match, distribution_type)，值为对应的差异矩阵 D_M。
#     """
#     matches = ['StarCraft', 'Tennis', 'Go', 'Badminton']
#     distributions = ['Uniform', 'PL', 'Normal']
#
#     fig, axes = plt.subplots(4, 3, figsize=(14, 14))  # 4 行 3 列子图，整体保持接近正方形
#     fig.suptitle("Difference Matrices Heatmaps", fontsize=18)
#
#     # 预先确定所有矩阵的最大最小值，保证色阶一致
#     all_values = np.concatenate([matrix.flatten() for matrix in difference_matrices.values()])
#     vmin, vmax = np.min(all_values), np.max(all_values)
#
#     # 绘制热度图
#     for i, match in enumerate(matches):
#         for j, dist in enumerate(distributions):
#             ax = axes[i, j]  # 选择对应子图
#             matrix = difference_matrices.get((match, dist))
#
#             # 绘制热度图，统一色阶范围
#             im = sns.heatmap(matrix, ax=ax, cmap='coolwarm', vmin=vmin, vmax=vmax,
#                              annot=False, cbar=False, square=True)  # square=True 确保正方形
#
#             ax.set_title(f"{match} - {dist}", fontsize=12)
#             ax.set_xlabel("Player")
#             ax.set_ylabel("Player")
#
#             # 设置正方形比例
#             ax.set_aspect('equal')
#
#     # 添加统一的 colorbar
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 右侧 colorbar
#     fig.colorbar(im.get_children()[0], cax=cbar_ax)
#
#     plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 留出右侧 colorbar 空间
#     plt.show()

def plot_theta(results, theta_values,distribution):
    """
    为每个比赛（match）绘制 Mean 随 Theta 变化的曲线，并标注最小值。

    参数:
    results: dict, 结构如下:
        {
            'StarCraft': {
                'Uniform': {'mean': [values], 'min': min_value, 'index': min_index},
                'PL': {'mean': [values], 'min': min_value, 'index': min_index},
                'Normal': {'mean': [values], 'min': min_value, 'index': min_index}
            },
            ...
        }
    theta_values: np.array, 存储 theta 的取值范围
    """

    matches = results.keys()
    distributions = distribution
    colors = ['b', 'g', 'r', 'y']  # 颜色列表

    plt.figure(figsize=(12, 8))

    for i, match in enumerate(matches):
        plt.subplot(2, 2, i + 1)  # 2行2列子图
        plt.title(f'{match} - D vs Theta')

        for j, dist in enumerate(distributions):
            if dist in results[match]:
                mean_values = results[match][dist]['mean']
                min_index = results[match][dist]['index']
                min_theta = theta_values[min_index]
                min_mean = results[match][dist]['min']

                # 画出曲线
                plt.plot(theta_values, mean_values, label=f"{dist} (θ={min_theta:.2f}, Min={min_mean:.2f})",
                         color=colors[j])

                # 标注最小值
                plt.scatter(min_theta, min_mean, color=colors[j], marker='o', s=50)
                # plt.text(min_theta, min_mean, f"({min_theta:.2f}, {min_mean:.2f})", fontsize=10,
                #          verticalalignment='bottom')

        plt.xlabel("Theta")
        plt.ylabel("D", rotation=0)
        plt.ylim(0, 8)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_R_difference(R_prime_result, R_result, matches, distributions):
    """
    绘制 R（原始结果）和 R'（不同分布模拟结果）的对比图，增强对比度，并使用指定颜色。

    参数：
    - R_prime_result: dict, 结构 {(match, distribution): R'}
    - R_result: dict, 结构 {match: R}
    - matches: list, 比赛种类
    - distributions: list, 分布种类
    """
    colors = ['b', 'g', 'r', 'y']  # 指定颜色
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 子图
    fig.suptitle("Comparison of R and R' Across Matches", fontsize=18)

    for i, match in enumerate(matches):
        ax = axes[i // 2, i % 2]  # 选择子图
        R = R_result[match]
        rank = np.arange(1, len(R) + 1)  # 生成 rank（从 1 开始）

        # 画 R 真实值（黑色虚线）
        ax.plot(rank, R, label="Original R", color="black", linestyle='dashed', linewidth=2)

        # 画 R'（三条曲线，分别对应三种分布）
        for j, dist in enumerate(distributions):
            R_prime = R_prime_result[(match, dist)]
            ax.plot(rank, R_prime, label=f"R' ({dist})", color=colors[j], linewidth=2)

        ax.set_title(match, fontsize=14)
        ax.set_xlabel("Rank")
        ax.set_ylabel("R Value")

        # 纵坐标从 0 到 0.9
        ax.set_ylim(0.35, 0.85)

        # 横坐标从 1 开始，每个 rank 显示
        ax.set_xticks(rank)
        ax.set_xticklabels(rank, rotation=0)  # 确保所有 rank 显示

        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()