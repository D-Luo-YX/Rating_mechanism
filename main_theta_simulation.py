import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# rating 函数的调用格式是 （人数， 矩阵维度系数, 是否使用NDCG）
# 比如 tennis_rating(False, 32, 1)
from theta_simulation.R_calculate import R_calculate
from theta_simulation.R_calculate import R_vector_calculate
# iteration, theta_value, winning_matrix, num_players, distribution_type
from theta_simulation.R_calculate import best_theta_simulation
from theta_simulation.R_calculate import best_theta_matrix_d
from theta_simulation.R_calculate import get_best_theta_matrix

from Matrix_Process.Badminton_rating import badminton_rating
from Matrix_Process.Tennis_rating import tennis_rating
from Matrix_Process.StarCraft_rating import scraft_rating
from Matrix_Process.Go_rating import go_rating

from Tournament.tournament import tournament_correlation

from plot_tools.plot_theta_simulation import plot_theta
from plot_tools.plot_theta_simulation import plot_difference_matrices
from plot_tools.plot_theta_simulation import plot_R_difference
from plot_tools.plot_tournament import plot_tournament_simulation, concat_images

def calculate_M(match_name, play_num, alpha, ndcg_flag):

    M = []
    if match_name == 'StarCraft':
        M = scraft_rating(play_num, alpha, ndcg_flag)
    if match_name == 'Tennis':
        M = tennis_rating(play_num, alpha, ndcg_flag)
    if match_name == 'Go':
        M = go_rating(play_num, alpha, ndcg_flag)
    if match_name == 'Badminton':
        M = badminton_rating(play_num, alpha, ndcg_flag)

    return M

def save_difference_matrices(difference_matrices, save_dir="result/difference_heat_matrix"):
    """
    将差异矩阵保存为 CSV 文件，按照 {match_name}/{distribution}.csv 结构存储。

    参数：
    - difference_matrices: dict, 结构为 {(match, distribution): matrix}
    - save_dir: str, 保存的根目录，默认 "result/difference_heat_matrix"
    """
    for (match, distribution), matrix in difference_matrices.items():
        # 确保目录存在
        match_dir = os.path.join(save_dir, match)
        os.makedirs(match_dir, exist_ok=True)

        # 保存 CSV 文件
        file_path = os.path.join(match_dir, f"{distribution}.csv")
        pd.DataFrame(matrix).to_csv(file_path, index=False, header=False)

        # print(f"✅ Saved: {file_path}")  # 打印保存路径，方便检查

if __name__ == '__main__':
    # setting parameters
    alpha = 1
    player_num = 32
    iteration = 10
    # and_one_flag = True
    and_one_flag = False

    NDCG_flag = False
    # NDCG_flag = True
    theta_values = np.arange(0, 2, 0.01)

    matches = ['StarCraft', 'Tennis', 'Go', 'Badminton']
    # matches = ['Tennis',]
    # distribution = ['Uniform', 'PL', 'Normal']
    distribution = ['Uniform', 'PL', 'Normal','MultiGaussian']
    # distribution = ['MultiGaussian']

    results = {match: {dist: {} for dist in distribution} for match in matches}

    #赛制的path
    correlations_tounament_simulation_dirpath = "tournament_result"
    if not os.path.exists(correlations_tounament_simulation_dirpath):
        os.makedirs(correlations_tounament_simulation_dirpath)
    real_path = os.path.join(correlations_tounament_simulation_dirpath,'real_data')
    simulation_path = os.path.join(correlations_tounament_simulation_dirpath,'simulation_data')
    # calculate D_mean; D_min; Theta_min_index
    # tennis_M = tennis_rating(player_num, alpha,False)
    # print(tennis_M.shape)

    ###########################################################
    ####################### best theta ########################
    ###########################################################
    for match in matches:
        temp_M = calculate_M(match, player_num, alpha, False)

        for distribution_type in distribution:
            temp_mean, temp_min, temp_index = R_calculate(iteration, theta_values, temp_M, player_num, distribution_type, match= match,and_one_flag= and_one_flag, NDCG_Flag= NDCG_flag)
            results[match][distribution_type] = {
                "mean": temp_mean,
                "min": temp_min,
                "index": temp_index
            }
    plot_theta(results, theta_values, distribution)


    ###########################################################
    ####################### 计算 R' 和 R #######################
    ###########################################################
    R_prime_result = {}
    R_result = {}

    # 计算 R'
    for match in matches:
        for dist in distribution:
            theta_ = theta_values[results[match][dist]['index']]
            temp_M = calculate_M(match, player_num, alpha, False)
            R_prime = best_theta_simulation(temp_M, theta_, dist, player_num, match=match, and_one_flag= and_one_flag)
            R_prime_result[(match, dist)] = R_prime

    # 计算 R   
    for match in matches:
        temp_M = calculate_M(match, player_num, alpha, False)

        if and_one_flag:
        # and_one
            R = R_vector_calculate(temp_M, player_num, len(temp_M))
        else:
        # no and_one
            R = R_vector_calculate(temp_M, player_num, len(temp_M)-1)

        R_result[match] = R

    # 画出R与R'的按位次波动的代码
    plot_R_difference(R_prime_result, R_result, matches, distribution)

    ###########################################################
    ############### 模拟数据和真实数据在赛制下的结果 ###############
    ###########################################################
    correlations = pd.DataFrame()
    tournament_iterations = 100 # 模拟次数
    for match in matches:
        real_winning_matrix = calculate_M(match, player_num, alpha, False)
        correlation_each_real = tournament_correlation(match, None, real_winning_matrix, tournament_iterations, 'Real Data')
        correlations = pd.concat([correlations, correlation_each_real], axis=0)
        for distribution_type in distribution:
            # 获取最佳 theta（使用之前计算的结果）
            best_theta = theta_values[results[match][distribution_type]['index']]

            # 获取最佳theta下的胜率模拟矩阵
            simulated_winning_matrix = get_best_theta_matrix(best_theta, distribution_type, player_num, match=match)

            # 在对应赛制和分布下，相关系数的值
            correlation_each_simulation = tournament_correlation(match, distribution_type, simulated_winning_matrix, tournament_iterations, 'Simulation Data')
            correlations = pd.concat([correlations, correlation_each_simulation], axis=0)
    correlations.to_csv(os.path.join(correlations_tounament_simulation_dirpath, "correlations_simulation.csv"), index=False)
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


    # # 计算差异热度图
    # difference_matrices = {}  # 存储差异矩阵
    # for match in matches:
    #     temp_M = calculate_M(match, player_num, alpha, False)  # 复用已有计算结果
    #
    #     for distribution_type in distribution:
    #         # 获取最佳 theta（使用之前计算的结果）
    #         best_theta = theta_values[results[match][distribution_type]['index']]
    #
    #         # 计算差异矩阵
    #         D_M = best_theta_matrix_d(temp_M, best_theta, distribution_type, player_num, match=match)
    #
    #         # 存储差异矩阵，方便后续调用
    #         difference_matrices[(match, distribution_type)] = D_M
    # # 保存差异热度矩阵
    # save_difference_matrices(difference_matrices)
    # # 画出差异热度图
    # plot_difference_matrices(difference_matrices, matches, distribution)