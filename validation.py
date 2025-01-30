import numpy as np
import pandas as pd
from Go_rating import go_rating
from Scraft_rating import scraft_rating
from Tennis_rating import tennis_rating
from Badminton_rating import badminton_rating

def E_vector_calculate(Winning_Matrix, player_num):
    """ 这个函数 用来得出每位选手战胜不如他的对手的概率
    参数：
    - Winning_Matrix: 当前需要处理的胜场矩阵
    """

    E = []
    N = [] #统计多少人交过手
    for i in range(player_num):
        n=0
        E_i = 0
        for j in range(i+1,player_num):
            Pij = Winning_Matrix[i, j]
            Pji = Winning_Matrix[j, i]
            if (Pij + Pji)>0.1:
                E_i += 1-Pij
                n+=1
        if n!=0:
            E_i=E_i/n
        E.append(1-E_i)
        N.append(n)
    return E


# def E_vector_calculate_ndcg(Winning_Matrix, player_num = 32):
#     """ 这个函数 用来得出每位选手战胜不如他的对手的概率，加入了权重因子
#     参数：
#     - Winning_Matrix: 当前需要处理的胜场矩阵
#     """

#     E = []
#     N = [] 
#     for i in range(player_num):
#         n = 0
#         E_i = 0
#         for j in range(i + 1, player_num):
#             if Winning_Matrix[i][j] > 0:
#                 n += 1
#                 weight = 1 / np.log(i + 2)  
#                 E_i += Winning_Matrix[i][j] * weight 
#         E.append(E_i / n if n > 0 else 0)
#         N.append(n)
#     return E

# Get Simulation Matrix M'
def strength_list(num_players, strengths_type, simulate_and_1_flag):
    if simulate_and_1_flag: simulate_num = int(num_players*2)
    else: simulate_num = int(num_players)
    if strengths_type == 'Uniform':
        # 均匀分布
        strengths = np.random.uniform(0, 1, simulate_num)

    #幂律分布
    elif strengths_type == 'PL':
        a = 0.5
        strengths = np.random.power(a, simulate_num)

    # 正态分布 重映射正态分布
    elif strengths_type == 'Normal':
        normal_data = np.random.randn(simulate_num)
        strengths = 1 / (1 + np.exp(-normal_data))

    strengths_df = pd.DataFrame(strengths, columns=["Strength"])
    strengths_df['Player'] = strengths_df.index

    sorted_strengths_df = strengths_df.sort_values(by="Strength", ascending=False).reset_index(drop=True)

    sorted_strengths = sorted_strengths_df['Strength'].values

    return sorted_strengths

# Get Simulation Matrix M'
def strength_list_ndcg(num_players, strengths_type, simulate_and_1_flag):
    """生成一组随机的实力值，用于模拟比赛，加入了权重因子"""
    if simulate_and_1_flag: simulate_num = int(num_players*2)
    else: simulate_num = int(num_players)
    if strengths_type == 'Uniform':
        # 均匀分布
        strengths = np.random.uniform(0, 1, simulate_num)

    #幂律分布
    elif strengths_type == 'PL':
        a = 0.5
        strengths = np.random.power(a, simulate_num)

    # 正态分布 重映射正态分布
    elif strengths_type == 'Normal':
        normal_data = np.random.randn(simulate_num)
        strengths = 1 / (1 + np.exp(-normal_data))

    strengths_df = pd.DataFrame(strengths, columns=["Strength"])
    strengths_df['Player'] = strengths_df.index

    sorted_strengths_df = strengths_df.sort_values(by="Strength", ascending=False).reset_index(drop=True)

    sorted_strengths = sorted_strengths_df['Strength'].values

    # # 引入NDCG思想
    discount_factors = 1 / (np.log(np.arange(1, simulate_num + 1) + 1))
    discounted_strengths = sorted_strengths * discount_factors

    return discounted_strengths

# def convert_to_n_and_1(prob_matrix, player_num = 32):

#     top_n = prob_matrix[:player_num, :player_num]
#     bottom_n = prob_matrix[player_num:, :]
#     avg_row_a1 = np.mean(bottom_n[:, :player_num], axis=0)
#     avg_col_a1 = np.mean(prob_matrix[:player_num, player_num:], axis=1)
#     avg_a1_a1 = np.mean(prob_matrix[player_num:, player_num:])
#     new_matrix = np.zeros((player_num+1, player_num+1))
#     new_matrix[:player_num, :player_num] = top_n
#     new_matrix[player_num, :player_num] = avg_row_a1
#     new_matrix[:player_num, player_num] = avg_col_a1
#     new_matrix[player_num, player_num] = avg_a1_a1

#     return new_matrix

# def ndcg_weights(rankings):
#     """计算 NDCG 权重"""
#     return 1 / np.log2(rankings + 2)  # +2 是因为 log2(1) 应该对应排名1

# def convert_to_33x33_ndcg(prob_matrix):
#     # 原始 32x32 的数据保留
#     top_32 = prob_matrix[:32, :32]

#     # 计算大于32排名选手的部分
#     bottom_32 = prob_matrix[32:, :]
#     bottom_ranks = np.arange(33, prob_matrix.shape[0] + 1)
#     weights = ndcg_weights(bottom_ranks)  # 对第33及以后排名计算 NDCG 权重

#     # 使用加权平均值代替简单平均
#     avg_row_33 = np.average(bottom_32[:, :32], axis=0, weights=weights[:len(bottom_32)])
#     avg_col_33 = np.average(prob_matrix[:32, 32:], axis=1, weights=weights[:prob_matrix.shape[1] - 32])

#     # 修正 avg_33_33 的计算
#     weights_2d = np.outer(weights, weights)  # 生成二维权重矩阵
#     avg_33_33 = np.sum(prob_matrix[32:, 32:] * weights_2d) / np.sum(weights_2d)

#     # 创建新的 33x33 矩阵
#     new_matrix = np.zeros((33, 33))
#     new_matrix[:32, :32] = top_32
#     new_matrix[32, :32] = avg_row_33
#     new_matrix[:32, 32] = avg_col_33
#     new_matrix[32, 32] = avg_33_33

#     return new_matrix

def calculate_simulation_matrix(simulated_strength, theta, num_players):

    temp_M = np.zeros((num_players, num_players), dtype=float)

    for i in range(num_players):
        for j in range(i, num_players):
            if i == j:
                P_i = 0
            else:
                # if (calculate_type == 1):
                P_i = (simulated_strength[i]) ** theta / ((simulated_strength[j]) ** theta + (simulated_strength[i]) ** theta)
                # else:
                #     strength_based_rate = (M_[i][j]) / ((M_[i][j]) + (M_[j][i]))
                #     random_rate = 0.5
                #     P_i = theta * strength_based_rate + (1 - theta) * random_rate
            temp_M[i, j] = P_i
            temp_M[j, i] = 1 - P_i
    # simulated_winning_matrix = convert_to_33x33(temp_M)
    # simulated_winning_matrix = convert_to_33x33_ndcg(temp_M)
    simulated_winning_matrix = temp_M
    return simulated_winning_matrix


# def simulation(matrix, theta_values, distribution_type = 'Uniform', num_players=32, simulate_and_1_flag = True):
#     D = []
#     matrix_n_n = matrix[:num_players, :num_players]
#     E = E_vector_calculate(matrix_n_n, num_players)

#     D_min = 10000
#     min_theta = 0
#     simulated_strength = strength_list_ndcg(num_players = num_players, strengths_type= distribution_type, simulate_and_1_flag = simulate_and_1_flag)
#     simulated_strength = simulated_strength[:num_players]
#     # simulated_strength = np.full(48, 0.5).tolist()
#     for theta in theta_values:
#         D_v = 0
#         simulated_winning_matrix = calculate_simulation_matrix(simulated_strength,theta)
#         # simulated_winning_matrix = standard_matrix(rows=33, cols=33)
#         E_simulated = E_vector_calculate(simulated_winning_matrix)
#         for i in range(num_players):
#             D_v += abs(E_simulated[i] - E[i])

#         D.append(D_v)
#         if D_v < D_min:
#             D_min = D_v
#             min_theta = theta
#     return D, min_theta , D_min
def simulation(matrix, theta_values, num_players, distribution_type):
    D = []
    D_min = 10000
    min_theta = 0

    matrix_n_n = matrix[:num_players, :num_players]
    E = E_vector_calculate(matrix_n_n, num_players)
    simulated_strength = strength_list(num_players, strengths_type = distribution_type, simulate_and_1_flag = simulate_and_1_flag)
    simulated_strength = simulated_strength[:num_players]
    for theta in theta_values:
        D_v = 0
        simulated_winning_matrix = calculate_simulation_matrix(simulated_strength, theta, num_players)
        E_simulated = E_vector_calculate(simulated_winning_matrix, num_players)
        for i in range(num_players):
            # D_v += abs((E_simulated[i] - E[i]) * (1/np.log2(i+2)))
            D_v += abs(E_simulated[i] - E[i]) 

        D.append(D_v)
        if D_v < D_min:
            D_min = D_v
            min_theta = theta
    return D, min_theta , D_min

def best_theta_simulation(num_players, simulate_and_1_flag, distribution_type, theta):
    D = []

    E_simulated_list = []

    for i in range(100):
        simulated_strength = strength_list(num_players=num_players, strengths_type=distribution_type,
                                           simulate_and_1_flag=simulate_and_1_flag)

        simulated_winning_matrix = calculate_simulation_matrix(simulated_strength, theta, num_players)
        # simulated_winning_matrix = standard_matrix(rows=33, cols=33)
        E_simulated = E_vector_calculate(simulated_winning_matrix, num_players)
        E_simulated_list.append(E_simulated)  # 添加到列表中

    # 计算 E_simulated 的平均值
    num_simulations = len(E_simulated_list)
    num_elements = len(E_simulated_list[0])

    E_simulated_avg = [sum(E_simulated_list[j][i] for j in range(num_simulations)) / num_simulations
                       for i in range(num_elements)]

    return E_simulated_avg

if __name__ == "__main__":
    # winning_matrix = go_rating()
    # D,_,_ = simulation(winning_matrix)
    # print(D)
    winning_matrix = tennis_rating()
    D, _, _ = simulation(winning_matrix)
    print(D)
    # winning_matrix = badminton_rating()
    # D, _, _ = simulation(winning_matrix)
    # print(D)
    # winning_matrix = scraft_rating()
    # D, _, _ = simulation(winning_matrix)
    # print(D)
