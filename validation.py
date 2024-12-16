import numpy as np
import pandas as pd
from Go_rating import go_rating
def E_vector_calculate(Winning_Matrix):
    """ 这个函数 用来得出每位选手战胜不如他的对手的概率
    参数：
    - Winning_Matrix: 当前需要处理的胜场矩阵
    """

    E = []
    N = [] #统计多少人交过手
    for i in range(33):
        n=0
        E_i = 0
        for j in range(i+1,33):
            Pij = Winning_Matrix[i, j]
            Pji = Winning_Matrix[j, i]
            if (Pij + Pji)>0.1:
                E_i += (1-Pij)
                n+=1
        if n!=0:
            E_i=E_i/n
        E.append(1-E_i)
        N.append(n)
    return E

# Get Simulation Matrix M'
def strength_list(num_players=48, strengths_type='Uniform'):
    if strengths_type == 'Uniform':
        # 均匀分布
        strengths = np.random.uniform(0, 1, num_players)

    #幂律分布
    elif strengths_type == 'PL':
        a = 0.5
        strengths = np.random.power(a, num_players)

    # 正态分布 重映射正态分布
    elif strengths_type == 'Normal':
        normal_data = np.random.randn(num_players)
        strengths = 1 / (1 + np.exp(-normal_data))

    strengths_df = pd.DataFrame(strengths, columns=["Strength"])
    strengths_df['Player'] = strengths_df.index

    sorted_strengths_df = strengths_df.sort_values(by="Strength", ascending=False).reset_index(drop=True)

    sorted_strengths = sorted_strengths_df['Strength'].values

    return sorted_strengths

def convert_to_33x33(prob_matrix):

    top_32 = prob_matrix[:32, :32]
    bottom_32 = prob_matrix[32:, :]
    avg_row_33 = np.mean(bottom_32[:, :32], axis=0)
    avg_col_33 = np.mean(prob_matrix[:32, 32:], axis=1)
    avg_33_33 = np.mean(prob_matrix[32:, 32:])
    new_matrix = np.zeros((33, 33))
    new_matrix[:32, :32] = top_32
    new_matrix[32, :32] = avg_row_33
    new_matrix[:32, 32] = avg_col_33
    new_matrix[32, 32] = avg_33_33

    return new_matrix

def calculate_simulation_matrix(simulated_strength,theta,num_players=48):

    temp_M = np.zeros((num_players, num_players), dtype=float)

    for i in range(num_players):
        for j in range(num_players):
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
    simulated_winning_matrix = convert_to_33x33(temp_M)
    return simulated_winning_matrix

def simulation(matrix,distribution_type = 'Uniform'):
    D = []
    E = E_vector_calculate(matrix)
    D_min = 10000
    min_theta = 0
    simulated_strength = strength_list(num_players=48, strengths_type= distribution_type)
    for theta in np.arange(0, 2, 0.01):
        D_v = 0
        simulated_winning_matrix = calculate_simulation_matrix(simulated_strength,theta)
        E_simulated = E_vector_calculate(simulated_winning_matrix)

        for i in range(32):
            D_v += abs(E_simulated[i] - E[i])

        D.append(D_v)
        if D_v < D_min:
            D_min = D_v
            min_theta = theta
    return D, min_theta , D_min

if __name__ == "__main__":
    winning_matrix = go_rating()
    simulation(winning_matrix)
