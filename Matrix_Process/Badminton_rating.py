import numpy as np
import pandas as pd
import csv
import os


def process_match_file(file_path, player_num = 32, alpha = 1):

    matrix_shap = int(player_num * alpha)
    win_matrix = np.zeros((matrix_shap+1, matrix_shap+1), dtype=int)

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("date"):
                continue

            try:
                parts = line.strip().split(",")
                _, player1, rank1, result, player2, rank2 = parts

                rank1 = int(rank1) if rank1.isdigit() and int(rank1) <= matrix_shap else (matrix_shap + 1)
                rank2 = int(rank2) if rank2.isdigit() and int(rank2) <= matrix_shap else (matrix_shap + 1)

                if result == "胜":
                    win_matrix[rank1 - 1, rank2 - 1] += 1
                elif result == "负":
                    win_matrix[rank2 - 1, rank1 - 1] += 1
            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")
                continue

    return win_matrix

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

# def convert_to_33x33_ndcg(matrix, player_num=32):
#     # 保留前 player_num x player_num 的部分
#     top_32 = matrix[:player_num, :player_num]
#
#     # 计算第 i 行在第归并列之后的损失因子加权平均
#     avg_col_last_modified = np.array([
#         np.sum([
#             matrix[i, j] / (np.log2(j - player_num + 2) + 1)
#             for j in range(player_num, matrix.shape[1])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ]) if np.any([
#             matrix[i, j] > 0 or matrix[j, i] > 0
#             for j in range(player_num, matrix.shape[1])
#         ]) else 0
#         for i in range(player_num)
#     ])
#
#     # 计算第 j 列在第归并行之后的损失因子加权平均
#     avg_row_last_modified = np.array([
#         np.sum([
#             matrix[i, j] / (np.log2(i - player_num + 2) + 1)
#             for i in range(player_num, matrix.shape[0])
#             if matrix[i, j] > 0 or matrix[j, i] > 0
#         ])  if np.any([
#             matrix[i, j] > 0 or matrix[j, i] > 0
#             for i in range(player_num, matrix.shape[0])
#         ]) else 0
#         for j in range(player_num)
#     ])
#
#     new_matrix = np.zeros((player_num + 1, player_num + 1))
#     new_matrix[:player_num, :player_num] = top_32
#     new_matrix[:player_num, player_num] = avg_col_last_modified
#     new_matrix[player_num, :player_num] = avg_row_last_modified
#     new_matrix[player_num, player_num] = 0
#
#     return new_matrix

def badminton_rating(player_num = 32, alpha = 1, ndcg_flag = False):

    input_file = 'Data/Badminton_data/badminton_games.txt'

    temp_matrix = process_match_file(input_file, player_num=player_num)
    result_matrix = normalize(temp_matrix)
    # if ndcg_flag == False:
    #     result_matrix = convert_to_33x33_ndcg(result_matrix)
    return result_matrix

if __name__ == "__main__":
    os.chdir("..")
    a = badminton_rating(False, 32 ,2)
    print(a)

    os.chdir("Matrix_Process")
    df = pd.DataFrame(a)
    df.to_csv("M_result/Badminton.csv", index=False, header=False)
