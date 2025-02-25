from matplotlib import rcParams
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import math

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
    拼接大图
    """
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(image_files) < 1:
        raise ValueError("文件夹中至少包含一张图片。")
    
    images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]
    width, height = images[0].size
    num_images = len(images)
    if num_images == 2:
        cols = 2
        rows = 1
    elif num_images == 8:
        cols = 4
        rows = 2
    # cols = int(math.ceil(num_images ** 0.5))  # 计算列数
    # rows = int(math.ceil(num_images / cols))  # 计算行数
    
    new_image = Image.new('RGB', (cols * width, rows * height))
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        new_image.paste(image, (col * width, row * height))
    new_image.save(os.path.join(image_folder, 'tournament.png'), dpi=(300, 300))


if __name__ == "__main__":
    import pandas as pd
    correlations = pd.read_csv("tournament_result/correlations_simulation.csv")
    correlations_tounament_simulation_dirpath = "tournament_result"
    distribution = ['Uniform', 'PL', 'Normal','MultiGaussian']

    real_path = os.path.join(correlations_tounament_simulation_dirpath,'real_data')
    simulation_path = os.path.join(correlations_tounament_simulation_dirpath,'simulation_data')
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    if not os.path.exists(simulation_path):
        os.makedirs(simulation_path)

    for correlation_type in ['Spearman', 'NDCG_Spearman']:
        for data_type in ['Simulation Data', 'Real Data']:
            for distribution_type in distribution:
                if data_type == 'Real Data':
                    plot_tournament_simulation(correlations, real_path, correlation_type, distribution_type, data_type)
                else:
                    plot_tournament_simulation(correlations, simulation_path, correlation_type, distribution_type, data_type)
    concat_images(real_path)
    concat_images(simulation_path)