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

from Tournament.do_tournament import tournament_correlation
from Tournament.coefficient import calculate_Spearman_coefficient
from Tournament.coefficient import calculate_ndcg_Spearman_coefficient
from Tournament.normalized_tounaments import robin_round, swiss_round, double_elimination_random, weighted_round_robin, rr_knockout, ladder_tournament

from plot_tools.plot_theta_simulation import plot_theta
from plot_tools.plot_theta_simulation import plot_difference_matrices
from plot_tools.plot_theta_simulation import plot_R_difference
from plot_tools.plot_tournament import plot_tournament_simulation, concat_images, plot_tournament_curve, concat_curve_images

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

def tournament_plot(type, correlation_types, data_types, correlations, distribution_types):
    """
    绘制赛制相关图片
        参数设置
        - type: 'curve' or 'bar'
        - correlation_types: ['Spearman', 'NDCG_Spearman']
        - data_types: ['Real Data', 'Simulation Data']
        - distribution_types: ['Uniform', 'PL', 'Normal','MultiGaussian']
    """
    # 绘制曲线图的代码
    if type == 'curve':
        # 筛选出绘制曲线的数据
        correlations = correlations[correlations['Parameter Type'] == 'rounds']
        real_path = os.path.join(correlations_tounament_simulation_dirpath,'Curve/real_data')
        simulation_path = os.path.join(correlations_tounament_simulation_dirpath,'Curve/simulation_data')
        if not os.path.exists(real_path):
            os.makedirs(real_path)
        if not os.path.exists(simulation_path):
            os.makedirs(simulation_path)

        for correlation_type in correlation_types:
            for data_type in data_types:
                for distribution_type in distribution_types:
                    if data_type == 'Real Data':
                        plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, real_path)
                    else:
                        plot_tournament_curve(correlations, correlation_type, distribution_type, data_type, simulation_path)
        concat_curve_images(real_path, 'real_data', 'curve')
        concat_curve_images(simulation_path, 'simulation_data', 'curve')

    # 绘制柱状图的代码
    elif type == 'bar':
        correlations = correlations[correlations['Parameter Type'] == 'finish_all_rounds']
        real_path = os.path.join(correlations_tounament_simulation_dirpath,'Bar/real_data')
        simulation_path = os.path.join(correlations_tounament_simulation_dirpath,'Bar/simulation_data')
        if not os.path.exists(real_path):
            os.makedirs(real_path)
        if not os.path.exists(simulation_path):
            os.makedirs(simulation_path)
        for correlation_type in correlation_types:
            for data_type in data_types:
                for distribution_type in distribution_types:
                    if data_type == 'Real Data':
                        plot_tournament_simulation(correlations, real_path, correlation_type, distribution_type, data_type)
                    else:
                        plot_tournament_simulation(correlations, simulation_path, correlation_type, distribution_type, data_type)
        concat_curve_images(real_path, 'real_data', 'bar')
        concat_curve_images(simulation_path, 'simulation_data', 'bar')

if __name__ == '__main__':
    # ---------- 设置参数 ----------
    alpha = 1
    player_num = 32 #
    iteration = 10  # 迭代次数
    and_one_flag = False    # 是否考虑多计算一名选手
    k = 7   # GMM 参数, 可选3，5，7，9，11

    NDCG_flag = False   # 是否使用NDCG计算结果，NDCG可以有效考虑排名靠前的选手
    theta_values = np.arange(0, 2, 0.01)    # 分布类型的参数

    matches = ['StarCraft', 'Tennis', 'Go', 'Badminton']    # 所有的比赛类型
    distribution = ['Uniform', 'PL', 'Normal','Zipf', 'MultiGaussian']    # 所有可能的实力分布类型

    results = {match: {dist: {} for dist in distribution} for match in matches} # 存储比赛的结果
    if not os.path.exists('result'):
        os.makedirs('result')

    # ----------- 绘制D随theta变化的曲线，衡量不同theta下仿真胜率矩阵$M'$和真实胜率矩阵$M$的差异 ----------
    for match in matches:
        temp_M = calculate_M(match, player_num, alpha, False)

        for distribution_type in distribution:
            temp_mean, temp_min, temp_index = R_calculate(iteration, theta_values, temp_M, player_num, distribution_type, match, k,and_one_flag= and_one_flag, NDCG_Flag= NDCG_flag)
            results[match][distribution_type] = {
                "mean": temp_mean,
                "min": temp_min,
                "index": temp_index
            }
    plot_theta(results, theta_values, distribution)


    # ----------------- 计算R和R' -----------------
    """"
        R:真实数据下的一系列选手被实力低于他的选手击败的概率
        R':模拟数据下的一系列选手被实力低于他的选手击败的概率
    """
    R_prime_result = {}
    R_result = {}

    # 计算 R'
    for match in matches:
        for dist in distribution:
            theta_ = theta_values[results[match][dist]['index']]
            temp_M = calculate_M(match, player_num, alpha, False)
            R_prime = best_theta_simulation(temp_M, theta_, dist, player_num, match, k, and_one_flag= and_one_flag)
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

    # 绘制R与R'的按位次波动的代码
    plot_R_difference(R_prime_result, R_result, matches, distribution)

    # ----------------- 赛制相关结果，包括真实数据R和模拟数据R' -----------------
    tournament_iterations = 1       # 赛制的模拟次数
    """ 绘制曲线图的类型，三个可选参数
        ----‘matches’:  以比赛场次控制
        ----‘rounds’:  以轮次数控制
        ----‘finish_all_rounds’:  以完成所有轮次控制"""
    turnament_curve_type = 'rounds' 
    print(f"当前赛制的类型为：{turnament_curve_type}")

    # 不同控制机制下参数的取值范围
    param_min = 1
    if turnament_curve_type == 'matches':
        param_max = player_num * (player_num - 1) // 2
    elif turnament_curve_type == 'rounds':
        param_max = player_num
    elif turnament_curve_type == 'finish_all_rounds':
        param_max = 100
    param_values = list(range(param_min, param_max+1))
    print(f"当前赛制的参数范围为：{param_min} - {param_max}")

    #赛制的path
    correlations_tounament_simulation_dirpath = f"result/tournament_result/{turnament_curve_type}/tournament_result_simulations={tournament_iterations}_parmmin={param_min}_prammax={param_max}"
    if not os.path.exists(correlations_tounament_simulation_dirpath):
        os.makedirs(correlations_tounament_simulation_dirpath)
    correlations = pd.DataFrame()
    correlations_path = os.path.join(correlations_tounament_simulation_dirpath, "correlations_tournament.csv")

    print(f"赛制相关结果的路径为：{correlations_path}")
    for match in matches:
        ### 真实数据下的结果
        # 获取真实胜率模拟矩阵
        real_winning_matrix = calculate_M(match, player_num, alpha, False)
        tournament_correlation(match, None, real_winning_matrix, tournament_iterations, 'Real Data', turnament_curve_type, param_values, correlations_path)

        # 模拟数据下的结果
        for distribution_type in distribution:
            # 获取最佳 theta（使用之前计算的结果）
            best_theta = theta_values[results[match][distribution_type]['index']]
            # 获取最佳theta下的胜率模拟矩阵
            simulated_winning_matrix = get_best_theta_matrix(best_theta, distribution_type, player_num, match, k)
            # 在对应赛制和分布下，相关系数的值
            tournament_correlation(match, distribution_type, real_winning_matrix, tournament_iterations, 'Simulation Data', turnament_curve_type, param_values, correlations_path)
            # tournament_correlation(match, distribution_type, real_winning_matrix, tournament_iterations, 'Simulation Data', 'finish_all_rounds', param_values, correlations_path)

    # 绘制图片
    correlations = pd.read_csv(correlations_path)

    tournament_plot('curve', ['Spearman'], ['Real Data', 'Simulation Data'], correlations, distribution)
    tournament_plot('curve', ['NDCG_Spearman'], ['Real Data', 'Simulation Data'], correlations, distribution)

    # tournament_plot('bar', ['Spearman'], ['Real Data', 'Simulation Data'], correlations, distribution)

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