import numpy as np
import pandas as pd
import csv
import os
import pandas as pd

### Converting the .txt to the Matrix
def process_match_file(input_file, output_file, player_num=64):
    matrix = np.zeros((player_num+1, player_num+1), dtype=int)

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

            rank_a = min(player_a_rank, player_num+1)
            rank_b = min(player_b_rank, player_num+1)

            if player_a_rank > player_num and player_b_rank > player_num:
                continue

            if player_a_score > player_b_score:
                matrix[rank_a - 1][rank_b - 1] += 1
            elif player_b_score > player_a_score:
                matrix[rank_b - 1][rank_a - 1] += 1

    return  matrix

def process_match_file_2(input_file, output_file, player_num=64):
    matrix = np.zeros((10000, 10000), dtype=int)

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


            if player_a_rank > player_num and player_b_rank > player_num:
                continue

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


def convert_matrix233(matrix, player_num=64):
    """Convert a larger matrix to a 33x33 matrix, skipping empty elements in calculations."""

    # 原始 32x32 部分保留
    top_32 = matrix[:player_num, :player_num]

    # 计算第 i 行在第 归并 列之后的非零元素统计
    avg_col_last_modified = np.array([
        np.mean([
            matrix[i, j]
            for j in range(player_num, matrix.shape[1])
            if matrix[i, j] > 0 or matrix[j, i] > 0
        ]) if np.any([matrix[i, j] > 0 or matrix[j, i] > 0 for j in range(player_num, matrix.shape[1])]) else 0
        for i in range(player_num)
    ])
    # 计算第 j 列在第 归并 行之后的非零元素统计
    avg_row_last_modified = np.array([
        np.mean([
            matrix[i, j]
            for i in range(player_num, matrix.shape[0])
            if matrix[i, j] > 0 or matrix[j, i] > 0  # 检查非零元素
        ]) if np.any([
            matrix[i, j] > 0 or matrix[j, i] > 0
            for i in range(player_num, matrix.shape[0])
        ]) else 0
        for j in range(player_num)
    ])

    # 创建新的 player_num + 1 矩阵
    new_matrix = np.zeros((player_num+1, player_num+1))
    new_matrix[:player_num, :player_num] = top_32
    new_matrix[:player_num, player_num] = avg_col_last_modified[:player_num]  # 修正为只取前 player_num 列 的结果
    new_matrix[player_num, :player_num] = avg_row_last_modified[:player_num]  # 修正为只取前 player_num 行 的结果
    new_matrix[player_num, player_num] = 0

    return new_matrix


def tennis_rating():
    input_file = 'Data/Tennis_data/match_data.txt'
    output_file = 'Data/Tennis_data/matrix.txt'
    # temp_matrix_1 = process_match_file(input_file, output_file)
    # winning_matrix_1 = normalize(temp_matrix_1)
    # print(winning_matrix_1)
    temp_matrix_2 = process_match_file_2(input_file, output_file)
    winning_matrix_2 = normalize(temp_matrix_2)
    transfer_matrix = convert_matrix233(winning_matrix_2)
    return transfer_matrix

if __name__ == "__main__":
    a = tennis_rating()
    print(a)
    csv_output_file = 'Data/Tennis_data/transfer_matrix.csv'
    pd.DataFrame(a).to_csv(csv_output_file, index=False, header=False)
    print(f"Matrix saved to {csv_output_file}")
    print(a[:,-1])