from matplotlib import rcParams
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import io

def plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, real_path):
    """
    绘制不同 Match 下的 Spearman 相关系数折线图。
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
        plt.grid(True)
        # 设置横轴范围为 1～35（显示数据 1～34，并留有边距），纵轴范围为 0～1
        plt.xlim(1, 35)
        plt.ylim(0, 1)
        # 显示 x 轴刻度：从 1 到 35，间隔为 5
        plt.xticks(np.arange(1, 35, 5))
        # 显示 y 轴刻度：从 -0.1 到 1.1，间隔为 0.1
        plt.yticks(np.arange(-0.1, 1.01, 0.1))
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
    metric_columns = [col for col in scores_result.columns if col not in ['Match', 'Distribution Type', 'Data Type'] and col.startswith(correlation_type)]
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
    # 为柱状图设置纵轴范围为0～1
    plt.ylim(0, 1)
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
    
    save_path = os.path.join(image_folder, 'tournament.png')
    new_image.save(save_path, dpi=(300, 300))
    return save_path

def concat_curve_images(image_folder, data_type, picture_name):
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
    
    save_path = os.path.join(image_folder, f'tournament_{picture_name}_{data_type}.png')
    new_image.save(save_path, dpi=(300, 300))
    return save_path

def create_legend_image(metric_columns, colors, markers, legend_title="Tournament"):
    """
    生成曲线图图例图片，返回一个 PIL.Image 对象。
    这里采用 marker 显示方式。
    """
    # 创建自定义图例的句柄
    handles = [Line2D([0], [0], marker=markers[i], color=colors[i],
                      linestyle='-', markersize=8, label=metric) 
               for i, metric in enumerate(metric_columns)]
    
    # 新建一个空白图，用于只显示图例
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(metric_columns), title=legend_title)
    fig.canvas.draw()

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    legend_image = Image.open(buf)
    plt.close(fig)
    return legend_image

def create_legend_image_bar(metric_columns, colors, legend_title="Tournament"):
    """
    生成柱状图图例图片，返回一个 PIL.Image 对象。
    这里采用颜色块显示方式。
    """
    handles = [mpatches.Patch(color=colors[i % len(colors)], label=metric_columns[i]) 
               for i in range(len(metric_columns))]
    
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(metric_columns), title=legend_title)
    fig.canvas.draw()

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    legend_image = Image.open(buf)
    plt.close(fig)
    return legend_image

def concat_image_with_legend(big_image_path, legend_image):
    """
    将大图和图例图像在垂直方向拼接，返回新的拼接后图像的 PIL.Image 对象。
    """
    big_img = Image.open(big_image_path)
    # 调整 legend 的宽度与大图一致
    new_width = big_img.width
    legend_ratio = new_width / legend_image.width
    new_legend_height = int(legend_image.height * legend_ratio)
    legend_img_resized = legend_image.resize((new_width, new_legend_height), Image.Resampling.LANCZOS)
    
    total_height = big_img.height + new_legend_height
    new_img = Image.new('RGB', (new_width, total_height), color='white')
    new_img.paste(big_img, (0, 0))
    new_img.paste(legend_img_resized, (0, big_img.height))
    return new_img

if __name__ == "__main__":
    correlation_types = ['Spearman']
    data_types = ['Real Data', 'Simulation Data']
    distribution_types = ['Uniform', 'PL', 'Normal', 'MultiGaussian']
    correlations_tounament_simulation_dirpath = "tournament_result_simulations=50_parmmin=1_prammax=34"
    correlations = pd.read_csv(os.path.join(correlations_tounament_simulation_dirpath, 'correlations_tournament.csv'))
    # 只保留Parameter从1到33的数据（后续横轴设置为1～34）
    correlations = correlations[correlations['Parameter'] <= 33]
    plot_type = 'curve'  # 可选择 'curve' 或 'bar'

    # 计算所有指标列（假设每个绘图函数中用到的指标列是一致的）
    # 例如以 'Spearman' 开头的所有列
    metric_columns = [col for col in correlations.columns if col.startswith("Spearman")]

    if plot_type == 'curve':
        # 绘制曲线图的代码
        correlations = correlations[correlations['Parameter Type'] == 'rounds']
        real_path = os.path.join(correlations_tounament_simulation_dirpath, 'Curve/real_data')
        simulation_path = os.path.join(correlations_tounament_simulation_dirpath, 'Curve/simulation_data')
        os.makedirs(real_path, exist_ok=True)
        os.makedirs(simulation_path, exist_ok=True)

        for correlation_type in correlation_types:
            for data_type in data_types:
                for distribution_type in distribution_types:
                    if data_type == 'Real Data':
                        plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, real_path)
                    else:
                        plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, simulation_path)
        # 拼接大图
        real_big_image_path = concat_curve_images(real_path, 'real_data', 'curve')
        simulation_big_image_path = concat_curve_images(simulation_path, 'simulation_data', 'curve')
        
    elif plot_type == 'bar':
        # 绘制柱状图的代码
        correlations = correlations[correlations['Parameter Type'] == 'finish_all_rounds']
        real_path = os.path.join(correlations_tounament_simulation_dirpath, 'Bar/real_data')
        simulation_path = os.path.join(correlations_tounament_simulation_dirpath, 'Bar/simulation_data')
        os.makedirs(real_path, exist_ok=True)
        os.makedirs(simulation_path, exist_ok=True)
        for correlation_type in correlation_types:
            for data_type in data_types:
                for distribution_type in distribution_types:
                    if data_type == 'Real Data':
                        plot_tournament_simulation(correlations, real_path, correlation_type, distribution_type, data_type)
                    else:
                        plot_tournament_simulation(correlations, simulation_path, correlation_type, distribution_type, data_type)
        # 拼接大图
        real_big_image_path = concat_curve_images(real_path, 'real_data', 'bar')
        simulation_big_image_path = concat_curve_images(simulation_path, 'simulation_data', 'bar')
    
    # 根据绘图类型选择对应的图例生成方式
    if plot_type == 'bar':
        # 对于柱状图，使用颜色块生成图例（无需 marker）
        # 注意这里的 colors 数组可以与 plot_tournament_simulation 中的颜色保持一致
        colors_bar = [
            '#41539F',  
            '#67589B', 
            '#3075B6', 
            '#3DB4E5', 
            '#E77E7F',
            '#FDCF1D',
        ]
        legend_img = create_legend_image_bar(metric_columns, colors_bar, legend_title="Tournament")
    else:
        # 对于曲线图，使用 marker 生成图例
        colors_curve = plt.cm.tab10(np.linspace(0, 1, len(metric_columns)))
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', 'P']
        legend_img = create_legend_image(metric_columns, colors_curve, markers, legend_title="Tournament")
    
    # 对于 Real Data 和 Simulation Data 的大图，分别添加图例并保存
    for big_image_path in [real_big_image_path, simulation_big_image_path]:
        new_img = concat_image_with_legend(big_image_path, legend_img)
        new_img.save(big_image_path, dpi=(300, 300))