import numpy as np
import pandas as pd
import os
from pathlib import Path
from Matrix_Process.Tennis_rating import tennis_rating

def R_vector_calculate(Winning_Matrix, player_num, calculate_number):
    """ 这个函数 用来得出每位选手战胜不如他的对手的概率
    参数：
    - Winning_Matrix: 当前需要处理的胜场矩阵
    - player_num: 返回的长度
    - calculate_number: 选手的平均胜率的计算长度
    比如计算32个人时，我们可能使用 32 * 2 + 1 作为calculate_number 然后返回前32
    """

    R = []
    N = [] #统计多少人交过手
    for i in range(calculate_number):
        n=0
        R_i = 0
        for j in range(i+1, calculate_number):
            Pij = Winning_Matrix[i, j]
            Pji = Winning_Matrix[j, i]
            if (Pij + Pji)>0.1:
                R_i += 1-Pij
                n+=1
        if n!=0:
            R_i = R_i/n
        R.append(1 - R_i)
        N.append(n)

    return_index =min(player_num,(calculate_number-1))
    return R[:return_index]

def strength_list(num_players, strengths_type):
    simulate_num = int(num_players)
    strengths = []
    if strengths_type == 'Uniform':
        # 均匀分布
        strengths = np.random.uniform(0, 1, simulate_num)

    #幂律分布
    elif strengths_type == 'PL':
        a = 1.5
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

def calculate_simulation_matrix(simulated_strength, theta, num_players):

    temp_M = np.zeros((num_players, num_players), dtype=float)

    for i in range(num_players):
        for j in range(i, num_players):
            if i == j:
                P_i = 0
            else:
                P_i = (simulated_strength[i]) ** theta / ((simulated_strength[j]) ** theta + (simulated_strength[i]) ** theta)

            temp_M[i, j] = P_i
            temp_M[j, i] = 1 - P_i

    simulated_winning_matrix = temp_M
    return simulated_winning_matrix

def simulation(matrix, theta_values, num_players, distribution_type, and_one_flag=True, NDCG_Flag = False):
    D = []
    D_min = 10000
    min_theta = 0
    if and_one_flag:
        calculate_length = len(matrix)
    else:
        calculate_length = len(matrix)-1
    # print(calculate_length)
    R = R_vector_calculate(matrix, num_players, calculate_length)

    simulated_strength = strength_list(calculate_length, strengths_type = distribution_type)
    simulated_strength = simulated_strength[:calculate_length]
    for theta in theta_values:
        D_v = 0
        simulated_winning_matrix = calculate_simulation_matrix(simulated_strength, theta, calculate_length)
        R_simulated = R_vector_calculate(simulated_winning_matrix, num_players, calculate_length)
        # 这里

        for i in range(len(R)):
            if NDCG_Flag:
                D_v += abs((R_simulated[i] - R[i]) * (1/np.log2(i+2)))
            else:
                D_v += abs(R_simulated[i] - R[i])

        D.append(D_v)
        if D_v < D_min:
            D_min = D_v
            min_theta = theta
    # print(f"R length: {len(R_simulated)}")
    # print(f"D length: {len(D)}")
    return D, min_theta , D_min

def best_theta_simulation(matrix, theta, distribution_type, num_players, and_one_flag=True):
    if and_one_flag:
        calculate_length = len(matrix)
    else:
        calculate_length = len(matrix)-1
    R_simulated_list = []

    for i in range(100):
        simulated_strength = strength_list(num_players=calculate_length, strengths_type=distribution_type)

        simulated_winning_matrix = calculate_simulation_matrix(simulated_strength, theta, calculate_length)
        R_simulated = R_vector_calculate(simulated_winning_matrix, num_players, calculate_length)
        R_simulated_list.append(R_simulated)  # 添加到列表中


    # 计算 E_simulated 的平均值
    num_simulations = len(R_simulated_list)
    num_elements = len(R_simulated_list[0])

    R_simulated_avg = [sum(R_simulated_list[j][i] for j in range(num_simulations)) / num_simulations
                       for i in range(num_elements)]
    # R = R_vector_calculate(matrix, num_players, calculate_length)
    # D_v = 0
    # for i in range(num_players):
    #     # D_v += abs((R_simulated[i] - R[i]) * (1/np.log2(i+2)))
    #     D_v += abs(R_simulated_avg[i] - R[i])
    #
    # print(D_v)
    return R_simulated_avg

def best_theta_matrix_d(matrix, theta, distribution_type, num_players, and_one_flag=True):
    D_M = []
    simulated_winning_matrix = np.zeros_like(matrix, dtype=float)
    if and_one_flag:
        calculate_length = len(matrix)
    else:
        calculate_length = len(matrix)-1
    for i in range(100):
        simulated_strength = strength_list(num_players=calculate_length, strengths_type=distribution_type)

        simulated_winning_matrix += calculate_simulation_matrix(simulated_strength, theta, calculate_length)

    D_M = matrix - simulated_winning_matrix/100
    D_M = D_M[:num_players,:num_players]

    return D_M

def save_d(match_name,  D_mean):
    save_path = Path(f"result/{match_name}")
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    # 保存 D_mean 到 CSV 文件
    df = pd.DataFrame({"D_mean": D_mean})
    df.to_csv(save_path.with_suffix(".txt"), index=False)


def R_calculate(iteration, theta_value, winning_matrix, num_players, distribution_type, and_one_flag=True, NDCG_Flag = False):
    D = []
    for i in range(iteration):
        temp_D, _, _ = simulation(winning_matrix, theta_value, num_players, distribution_type, and_one_flag=and_one_flag, NDCG_Flag=NDCG_Flag)
        D.append(temp_D)

    D_mean = np.mean(np.array(D), axis=0)

    d_min = np.min(D_mean)
    min_index = np.argmin(D_mean)

    return D_mean, d_min, min_index

if __name__ == "__main__":
    current_path = Path.cwd()

    iteration = 100
    os.chdir("..")
    D = []
    theta_values = np.arange(0, 2, 0.01)

    winning_matrix = tennis_rating(32,2,False)
    # D, min, d_min = simulation(winning_matrix, theta_values, 32, "Uniform")
    for i in range(iteration):
        temp_D, _, _ = simulation(winning_matrix, theta_values, 32, "Uniform")
        D.append(temp_D)

    D_mean = np.mean(np.array(D), axis=0)

    d_min = np.min(D_mean)
    min_index = np.argmin(D_mean)
    theta_at_min = theta_values[min_index]

    print(D_mean)
    print(d_min)
    print(theta_at_min)
    save_d("Tennis", D_mean)
