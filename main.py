import numpy as np
import os
import matplotlib.pyplot as plt
import random

from sympy import false

from Go_rating import go_rating
from Tennis_rating import tennis_rating
from Badminton_rating import badminton_rating
from Scraft_rating import  scraft_rating
from validation import simulation
from plot_E import plot_single_match

def save_result_to_txt(data_list,save_path):
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(save_path, "w", encoding="utf-8") as file:
        for item in data_list:
            file.write(f"{item}\n")
    return 0

def read_file(filename):
# Open the file in read mode ('r')
    with open(filename, 'r') as file:
        # Read each line, strip newline characters, and convert to the appropriate type
        number_list = [float(line.strip()) for line in file]
    return number_list

def plot_function(filename, theta_values):
    # Read data
    Uniform = read_file(f'./Result/{filename}/Uniform.txt')
    PL = read_file(f'./Result/{filename}/PL.txt')
    Normal = read_file(f'./Result/{filename}/Normal.txt')

    plt.figure(figsize=(10, 6))

    # Plot distributions
    plt.plot(theta_values, Uniform, linestyle='--', color='r', label='Uniform')
    plt.plot(theta_values, PL, linestyle='--', color='blue', label='PowerLaw')
    plt.plot(theta_values, Normal, linestyle='--', color='gray', label='Normal')

    # Find and mark minimum values
    def mark_minima(data, color, label):
        min_value = np.min(data)
        min_indices = np.where(data == min_value)[0]
        theta_min = theta_values[min_indices[0]]
        for idx in min_indices:
            plt.axvline(x=theta_values[idx], color=color, linestyle=':', alpha=0.7)    
            plt.scatter(theta_values[idx], min_value, color=color, label=f'theta={theta_min:.2f}' if idx == min_indices[0] else "")
        return theta_min
    Uniform_min = mark_minima(Uniform, 'r', 'Uniform')
    PL_min = mark_minima(PL, 'blue', 'PowerLaw')
    Normal_min = mark_minima(Normal, 'gray', 'Normal')

    # Add labels and legend
    plt.xlabel('theta')
    plt.ylabel('Difference(D)')
    plt.title(f'Comparison of Different Distributions in {filename} match')
    plt.legend()

    plt.grid(True)
    # plt.show()
    plt.savefig(f'./figure_jxs/{filename}_theta_D.png')
    return round(Uniform_min, 2), round(PL_min, 2), round(Normal_min, 2)

def standard_matrix(rows=3, cols=3):
    """
    Generates a matrix with the given dimensions, filled with the value 0.5.

    Parameters:
        rows (int): Number of rows in the matrix (default is 3).
        cols (int): Number of columns in the matrix (default is 3).

    Returns:
        numpy.ndarray: A matrix filled with 0.5.
    """
    return np.full((rows, cols), 0.5)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    theta_values = np.arange(0, 2, 0.01)
    winning_matrix = []
    match_set = ['go','tennis','badminton','scraft']
    results = []

    save_path = 'Result' # 保存结果txt文件的路径
    save_figure_path = 'figure_jxs' # 保存结果图片的路径

    # 如果只保留前32个选手，则rating_num_32=True
    rating_num_32 = True
    num_players = 32
    simulate_and_1_flag=false

    for i in range(len(match_set)):
        if not os.path.exists(os.path.join(save_path, match_set[i])):
            os.makedirs(os.path.join(save_path, match_set[i]))
        if not os.path.exists(save_figure_path):
            os.makedirs(save_figure_path)

    #胜负矩阵
    winning_matrix = {
    match: globals().get(f"{match}_rating", lambda x: None)(rating_num_32)
    for match in match_set
    }

    winning_matrix['Random'] = standard_matrix(rows=33, cols=33)

    match_to_matrix = {
    'go': winning_matrix['go'],
    'tennis': winning_matrix['tennis'],
    'badminton': winning_matrix['badminton'],
    'scraft': winning_matrix['scraft'],
    'random': winning_matrix['Random']
    }

    for match in match_set:
        match_name = match
        winning_matrix = match_to_matrix[match_name]

        strength = ['Normal','Uniform','PL']

        all_D = []
        for strength_type in strength:
            for iteration in range(100):
                # D, min_theta , D_min = simulation(winning_matrix,distribution_type=strength_type)#三种分布方式 'Uniform','PL','Normal' 分别代表均匀分布，幂律分布，正态分布。
                D, _, _ = simulation(winning_matrix, theta_values, rating_num_32, num_players, simulate_and_1_flag, distribution_type=strength_type)  # 三种分布方式 'Uniform','PL','Normal' 分别代表均匀分布，幂律分布，正态分布。
                all_D.append(D)
            all_D_array = np.array(all_D)
            D_mean = np.mean(all_D_array, axis=0)
            save_result_to_txt(D_mean, os.path.join(save_path, match_name, f"{strength_type}.txt"))

        # 绘制theta-D图
        uniform_min, PL_min, Normal_min = plot_function(match_name, theta_values)
        print(uniform_min, PL_min, Normal_min)
    
        results.append([match_name, uniform_min, PL_min, Normal_min])
        
        plot_single_match(match_name, uniform_min, PL_min, Normal_min, match_to_matrix, save_figure_path, num_players, simulate_and_1_flag)

    results_array = np.array(results, dtype=object)  # 使用 dtype=object 以兼容字符串列
    header = "Name Uniform_Min PL_Min Normal_Min"
    np.savetxt("Best.txt", results_array, fmt="%s", header=header, comments="")