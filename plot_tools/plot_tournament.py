from matplotlib import rcParams
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd

def plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, real_path):
    """
    绘制不同 Match 下的 Spearman 相关系数折线图。
    
    参数：
      - correlations: DataFrame，包含所有数据，必须包含以下列：
            "Match", "Distribution Type", "Data Type", "Parameter" 以及以 correlation_type 开头的指标列
      - correlation_type: 字符串，例如 "Spearman"，用于筛选指标列（取所有列名以该字符串开头）
      - distribution_type: 字符串，用于筛选数据中 "Distribution Type" 列的值（当 data_type 不为 "Real Data" 时使用）
      - data_type: 字符串，数据类型（例如 "Real Data" 或 "Simulation"）；筛选 "Data Type" 列必须一致
      - real_path: 字符串，保存图像的文件夹路径
    """
    # 先筛选出 Data Type 相同的数据
    df = correlations[correlations["Data Type"] == data_type].copy()
    # 如果 data_type 不为 "Real Data"，则还需筛选 Distribution Type
    if data_type != "Real Data":
        df = df[df["Distribution Type"] == distribution_type]
    
    if df.empty:
        print("筛选后没有符合条件的数据！")
        return

    # 获取所有以 correlation_type 开头的指标列
    metric_columns = [col for col in df.columns if col.startswith(correlation_type)]
    if not metric_columns:
        print("没有找到以 {} 开头的指标列。".format(correlation_type))
        return

    # 生成颜色和 marker 样式列表
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_columns)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', 'P']

    # 对每个不同的 Match 分组绘制图形
    matches = df["Match"].unique()
    for match in matches:
        sub_df = df[df["Match"] == match].copy()
        # 按 Parameter 排序
        sub_df = sub_df.sort_values(by="Parameter")
        x = sub_df["Parameter"]

        plt.figure(figsize=(10, 6))
        # 对于每个指标列绘制一条曲线
        for i, col in enumerate(metric_columns):
            y = sub_df[col]
            plt.plot(x, y, marker=markers[i % len(markers)], color=colors[i],
                     linestyle='-', label=col)
        
        plt.xlabel("Parameter")
        plt.ylabel(f"{correlation_type} Coefficient")
        plt.title(f"{match} - {data_type}" + (f" - {distribution_type}" if data_type != "Real Data" else ""))
        plt.legend(title="Tournament")
        plt.grid(True)
        plt.tight_layout()
        
        # 构造保存文件名
        safe_match = "".join([c if c.isalnum() or c in "-_" else "_" for c in str(match)])
        if data_type == "Real Data":
            filename = f"{safe_match}_{data_type}_{correlation_type}.png"
        else:
            filename = f"{safe_match}_{data_type}_{correlation_type}_{distribution_type}.png"
        save_file = os.path.join(real_path, filename)
        plt.savefig(save_file)
        plt.close()
        print(f"图像已保存至：{save_file}")


def plot_tournament_simulation(scores_result, result_path, correlation_type, distribution_type, data_type):
    """
    绘制 scores_result 的柱状图。
    """    
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['axes.unicode_minus'] = False

    match_types = scores_result['Match'].unique()
    scores_result = scores_result[scores_result['Data Type'] == data_type]
    if data_type == 'Simulation Data':
        scores_result = scores_result[scores_result['Distribution Type'] == distribution_type]

    # 以 correlation_type 开头的列
    metric_columns = [col for col in scores_result.columns if col != 'Match' and col != 'Distribution Type'  and col != 'Data Type' and col.startswith(correlation_type)]
    num_metrics = len(metric_columns)
    
    bar_width = 0.2  # 柱子宽度
    gap_between_groups = 0.5  # 各个match区域之间的间隔
    index = np.arange(len(match_types)) * (num_metrics * bar_width + gap_between_groups) # 每个match区域的横坐标位置

    colors = [
        '#41539F',  
        '#67589B', 
        '#3075B6', 
        '#3DB4E5', 
        '#E77E7F',
        '#FDCF1D',
    ]
    
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(metric_columns):
        values = [scores_result[scores_result['Match'] == match][col].mean() for match in match_types]
        plt.bar(index + i * bar_width, values, bar_width, label=col, color=colors[i % len(colors)])
    
    plt.xlabel('')
    plt.ylabel('Coefficient Value')
    if data_type == 'Simulation Data':
        plt.title(f'{correlation_type} of {data_type} ({distribution_type})')
    else:
        plt.title(f'{correlation_type} of {data_type}')

    plt.xticks(index + (num_metrics - 1) * bar_width / 2, match_types)
    plt.legend(title="Tournament", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.tight_layout()
    if data_type == 'Simulation Data':
        plt.savefig(os.path.join(result_path, f'{correlation_type}_{data_type}_{distribution_type}.png'))
    else:
        plt.savefig(os.path.join(result_path, f'{correlation_type}_{data_type}.png'))
    plt.close()


def concat_images(image_folder):
    """
    拼接大图，按照图片文件最后一个_后的单词排序
    """
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(image_files) < 1:
        raise ValueError("文件夹中至少包含一张图片。")
    
    # 按照文件名最后一个'_'之后的部分排序
    image_files.sort(key=lambda f: f.split('_')[-1])
    
    images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]
    width, height = images[0].size
    num_images = len(images)
    
    # 根据图片数量计算行列数
    if num_images == 2:
        cols = 2
        rows = 1
    elif num_images == 8:
        cols = 4
        rows = 2
    else:
        cols = int(math.ceil(num_images ** 0.5))  # 计算列数
        rows = int(math.ceil(num_images / cols))  # 计算行数
    
    new_image = Image.new('RGB', (cols * width, rows * height))
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        new_image.paste(image, (col * width, row * height))
    
    new_image.save(os.path.join(image_folder, 'tournament.png'), dpi=(300, 300))

def concat_curve_images(image_folder, data_type):
    """
    拼接大图，按照图片文件名称的字典序排序。
    """
    # 筛选出图片文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(image_files) < 1:
        raise ValueError("文件夹中至少包含一张图片。")
    
    # 直接按照文件名进行排序
    image_files.sort()
    
    images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]
    width, height = images[0].size
    num_images = len(images)
    
    # 根据图片数量计算行列数
    if num_images == 2:
        cols = 2
        rows = 1
    elif num_images == 8:
        cols = 4
        rows = 2
    else:
        cols = int(math.ceil(num_images ** 0.5))  # 计算列数
        rows = int(math.ceil(num_images / cols))  # 计算行数
    
    new_image = Image.new('RGB', (cols * width, rows * height))
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        new_image.paste(image, (col * width, row * height))
    
    new_image.save(os.path.join(image_folder, f'tournament_{data_type}.png'), dpi=(300, 300))

if __name__ == "__main__":
    # correlations = pd.read_csv("tournament_result/correlations_simulation.csv")
    # correlations_tounament_simulation_dirpath = "tournament_result"
    # distribution = ['Uniform', 'PL', 'Normal','MultiGaussian']

    # real_path = os.path.join(correlations_tounament_simulation_dirpath,'real_data')
    # simulation_path = os.path.join(correlations_tounament_simulation_dirpath,'simulation_data')
    # if not os.path.exists(real_path):
    #     os.makedirs(real_path)
    # if not os.path.exists(simulation_path):
    #     os.makedirs(simulation_path)

    # for correlation_type in ['Spearman', 'NDCG_Spearman']:
    #     for data_type in ['Simulation Data', 'Real Data']:
    #         for distribution_type in distribution:
    #             if data_type == 'Real Data':
    #                 plot_tournament_simulation(correlations, real_path, correlation_type, distribution_type, data_type)
    #             else:
    #                 plot_tournament_simulation(correlations, simulation_path, correlation_type, distribution_type, data_type)
    # concat_images(real_path)
    # concat_images(simulation_path)

    correlations = pd.read_csv("tournament_result_simulations=20_param=rounds_parmmin=1_prammax=40/correlations_tournament.csv")
    correlations_tounament_simulation_dirpath = "tournament_result_simulations=20_param=rounds_parmmin=1_prammax=40"
    distribution = ['Uniform', 'PL', 'Normal','MultiGaussian']

    real_path = os.path.join(correlations_tounament_simulation_dirpath,'real_data')
    simulation_path = os.path.join(correlations_tounament_simulation_dirpath,'simulation_data')
 
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    if not os.path.exists(simulation_path):
        os.makedirs(simulation_path)
    for correlation_type in ['Spearman']:
        for data_type in ['Simulation Data', 'Real Data']:
            for distribution_type in distribution:
                if data_type == 'Real Data':
                    plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, real_path)
                else:
                    plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, simulation_path)
    concat_curve_images(real_path, 'real_data')
    concat_curve_images(simulation_path, 'simulation_data')