import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
from pathlib import Path
def process_match_file(input_file, player_num=32, alpha =1):
    # matrix = np.zeros((10000, 10000), dtype=int)
    matrix_shape = int(player_num * alpha)

    matrix = np.zeros(shape=(matrix_shape+1, matrix_shape+1))

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip().split('\t')
            try:
                player_a_rank = int(data[2].strip())
                player_b_rank = int(data[3].strip())
                player_a_score = float(data[4].strip())
                player_b_score = float(data[5].strip())
            except ValueError:
                print(f"数据格式错误，跳过: {line}")
                continue

            if player_a_rank == player_b_rank:
                # print(f"跳过排名相同的比赛记录: A_rank={player_a_rank}, B_rank={player_b_rank}")
                continue

            # if player_a_rank > matrix_shape and player_b_rank > matrix_shape:
            #     continue
            # elif player_a_rank < matrix_shape and player_b_rank > matrix_shape:
            #     player_b_rank = matrix_shape+1
            # elif player_a_rank > matrix_shape and player_b_rank < matrix_shape:
            #     player_a_rank = matrix_shape+1

            if player_a_rank > matrix_shape: player_a_rank = matrix_shape + 1
            if player_b_rank > matrix_shape: player_b_rank = matrix_shape + 1

            if player_a_score > player_b_score:
                matrix[player_a_rank - 1][player_b_rank - 1] += 1
            elif player_b_score > player_a_score:
                matrix[player_b_rank - 1][player_a_rank - 1] += 1

    return  matrix

def normalize(M):
    """Normalize the matrix M so that Mij = Mij / (Mij + Mji)."""
    normalized_M = np.zeros_like(M, dtype=float)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i != j:
                total = M[i, j] + M[j, i]
                if total > 0:
                    normalized_M[i, j] = M[i, j] / total
                    normalized_M[j, i] = M[j, i] / total
    return normalized_M

# def convert_matrix_n_and_1(matrix, player_num=32):
#     """Convert a larger matrix to a 33x33 matrix, skipping empty elements in calculations."""

#     # 原始 32x32 部分保留
#     top_32 = matrix[:player_num, :player_num]

#     # 计算第 i 行在第 归并 列之后的非零元素统计
#     avg_col_last_modified = np.array([
#         np.mean([
#             matrix[i, j]
#             for j in range(player_num, matrix.shape[1])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ]) if np.any([matrix[i, j] > 0 or matrix[j, i] > 0 for j in range(player_num, matrix.shape[1])]) else 0
#         for i in range(player_num)
#     ])
#     # 计算第 j 列在第 归并 行之后的非零元素统计
#     avg_row_last_modified = np.array([
#         np.mean([
#             matrix[i, j]
#             for i in range(player_num, matrix.shape[0])
#             if matrix[i, j] > 0 or matrix[j, i] > 0  # 检查非零元素
#         ]) if np.any([
#             matrix[i, j] > 0 or matrix[j, i] > 0
#             for i in range(player_num, matrix.shape[0])
#         ]) else 0
#         for j in range(player_num)
#     ])

#     # 创建新的 player_num + 1 矩阵
#     new_matrix = np.zeros((player_num+1, player_num+1))
#     new_matrix[:player_num, :player_num] = top_32
#     new_matrix[:player_num, player_num] = avg_col_last_modified[:player_num]  # 修正为只取前 player_num 列 的结果
#     new_matrix[player_num, :player_num] = avg_row_last_modified[:player_num]  # 修正为只取前 player_num 行 的结果
#     new_matrix[player_num, player_num] = 0
#     return new_matrix

def convert_to_33x33_ndcg(matrix, player_num=32):
    # 保留前 player_num x player_num 的部分
    top_32 = matrix[:player_num, :player_num]

    # 计算第 i 行在第归并列之后的损失因子加权平均
    avg_col_last_modified = np.array([
        np.sum([
            matrix[i, j] / (np.log2(j - player_num + 2))
            for j in range(player_num, matrix.shape[1])
            if matrix[i, j] > 0 or matrix[j, i] > 0
        ]) if np.any([
            matrix[i, j] > 0 or matrix[j, i] > 0
            for j in range(player_num, matrix.shape[1])
        ]) else 0
        for i in range(player_num)
    ])

    # 计算第 j 列在第归并行之后的损失因子加权平均
    avg_row_last_modified = np.array([
        np.sum([
            matrix[i, j] / (np.log2(i - player_num + 2))
            for i in range(player_num, matrix.shape[0])
            if matrix[i, j] > 0 or matrix[j, i] > 0
        ])  if np.any([
            matrix[i, j] > 0 or matrix[j, i] > 0
            for i in range(player_num, matrix.shape[0])
        ]) else 0
        for j in range(player_num)
    ])

    new_matrix = np.zeros((player_num + 1, player_num + 1))
    new_matrix[:player_num, :player_num] = top_32
    new_matrix[:player_num, player_num] = avg_col_last_modified 
    new_matrix[player_num, :player_num] = avg_row_last_modified 
    new_matrix[player_num, player_num] = 0 

    return new_matrix

def tennis_rating(player_num=32, alpha =1, ndcg_flag = True):

    input_file = 'Data/Tennis_data/match_data.txt'
    output_file = 'Data/Tennis_data/matrix.txt'
    temp_matrix_1 = process_match_file(input_file, player_num=player_num, alpha =alpha)
    result_matrix = normalize(temp_matrix_1)

    # temp_matrix_1 = process_match_file(input_file, output_file)
    # winning_matrix_1 = normalize(temp_matrix_1)
    # print(winning_matrix_1)
    # temp_matrix_2 = process_match_file_2(input_file, output_file)
    # if ndcg_flag == False:
    #     result_matrix = convert_to_33x33_ndcg(temp_matrix_2)
    result_matrix = normalize(result_matrix)

    return result_matrix

if __name__ == "__main__":
    os.chdir('..')


    a = tennis_rating(32,2,False)
    print(a)
    print(a.shape)

    os.chdir("Matrix_Process")
    df = pd.DataFrame(a)
    df.to_csv("M_result/Tennis.csv", index=False, header=False)