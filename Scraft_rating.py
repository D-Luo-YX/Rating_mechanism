import os
import numpy as np


def normalize(M):
    """Normalize the matrix M so that Mij = Mij / (Mij + Mji)."""
    normalized_M = np.zeros_like(M, dtype=float)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i != j:
                total = M[i, j] + M[j, i]
                if total > 0:
                    normalized_M[i, j] = M[i, j] / total
    return normalized_M

def convert_to_33x33_ndcg(matrix, player_num=32):
    # 保留前 player_num x player_num 的部分
    top_32 = matrix[:player_num, :player_num]

    # 计算第 i 行在第归并列之后的损失因子加权平均
    avg_col_last_modified = np.array([
        np.sum([
            matrix[i, j] / (np.log2(j - player_num + 2) + 1)
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
            matrix[i, j] / (np.log2(i - player_num + 2) + 1)
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

# def convert_to_33x33_ndcg(matrix, player_num=32):
#     # 保留前 player_num x player_num 的部分
#     top_32 = matrix[:player_num, :player_num]

#     # 计算第 i 行在第归并列之后的损失因子加权平均
#     avg_col_last_modified = np.array([
#         np.sum([
#             matrix[i, j] / (np.log2(j - player_num + 2) + 1)
#             for j in range(player_num, matrix.shape[1])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ]) / np.sum([
#             1 / (np.log2(j - player_num + 2) + 1)
#             for j in range(player_num, matrix.shape[1])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ]) if np.any([
#             matrix[i, j] > 0 or matrix[j, i] > 0
#             for j in range(player_num, matrix.shape[1])
#         ]) else 0
#         for i in range(player_num)
#     ])

#     # 计算第 j 列在第归并行之后的损失因子加权平均
#     avg_row_last_modified = np.array([
#         np.sum([
#             matrix[i, j] / (np.log2(i - player_num + 2) + 1)
#             for i in range(player_num, matrix.shape[0])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ]) / np.sum([
#             1 / (np.log2(i - player_num + 2) + 1)
#             for i in range(player_num, matrix.shape[0])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ]) if np.any([
#             matrix[i, j] > 0 or matrix[j, i] > 0
#             for i in range(player_num, matrix.shape[0])
#         ]) else 0
#         for j in range(player_num)
#     ])

#     new_matrix = np.zeros((player_num + 1, player_num + 1))
#     new_matrix[:player_num, :player_num] = top_32
#     new_matrix[:player_num, player_num] = avg_col_last_modified 
#     new_matrix[player_num, :player_num] = avg_row_last_modified 
#     new_matrix[player_num, player_num] = 0 

#     return new_matrix


def scraft_rating(rating_num_32, player_num = 32):
    files = os.listdir('Data/Scraft_data/data/')
    M = np.zeros((101, 101))
    for item in files:
        M += np.load(f'Data/Scraft_data/data/{item}')
    result_matrix = np.zeros((player_num+1, player_num+1))
    result_matrix[:player_num, :player_num] = M[:player_num, :player_num]
    row_sums = M[:player_num, player_num:].sum(axis=1)
    result_matrix[:player_num, player_num] = row_sums

    col_sums = M[player_num:, :player_num].sum(axis=0)
    result_matrix[player_num, :player_num] = col_sums
    result_matrix = normalize(result_matrix)
    if rating_num_32 == False:
        result_matrix = convert_to_33x33_ndcg(result_matrix)
    return result_matrix



if __name__ == '__main__':
    a = scraft_rating()
    print(a)
    print(len(a))