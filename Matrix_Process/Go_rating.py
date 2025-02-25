import numpy as np
import pandas as pd
import csv
import os
from collections import defaultdict

def read_rankings(filename):
    rank_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            rank_list.append(row[1])
    return rank_list


def single_matrix(filename, player_num = 32, alpha = 1):

    matrix_shap = int(player_num * alpha) # M的尺寸
    if matrix_shap > 64:
        raise Exception("Matrix shape cannot over 64")


    rank_list_temp_ = read_rankings(f'Data/Go_64_data/{filename}/player_rankings.txt')
    rank_list_temp = rank_list_temp_[:matrix_shap]

    win_matrix = np.zeros((matrix_shap+1, matrix_shap+1), dtype=int) # 启用增广矩阵

    input_file = f'Data/Go_data/{filename}/player_games.txt'
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                parts = line.strip().split(", ")
                data = {key_value.split(": ")[0]: key_value.split(": ")[1] for key_value in parts}

                player = data.get("Player")
                opponent = data.get("Opponent")
                result = data.get("Result")
            except Exception:
                continue

            if player is not None and opponent is not None:
                if player in rank_list_temp:
                    player_idx = rank_list_temp.index(player)
                else:
                    player_idx = None

                if opponent in rank_list_temp:
                    opponent_idx = rank_list_temp.index(opponent)
                else:
                    opponent_idx = None

                if player_idx is not None and opponent_idx is not None:
                    if result == "胜":
                        win_matrix[player_idx, opponent_idx] += 1
                    elif result == "负":
                        win_matrix[opponent_idx, player_idx] += 1
                elif player_idx is not None:
                    if result == "胜":
                        win_matrix[player_idx, matrix_shap] += 1
                    elif result == "负":
                        win_matrix[matrix_shap, player_idx] += 1
                elif opponent_idx is not None:
                    if result == "负":
                        win_matrix[opponent_idx, matrix_shap] += 1
                    elif result == "胜":
                        win_matrix[matrix_shap, opponent_idx] += 1
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


def single_year(player_num=32, alpha = 1):
    begin_year = 1980
    M_temp = single_matrix(f'{begin_year}-01-01_to_{begin_year + 1}-01-01',player_num=player_num, alpha=alpha)
    normalized_single_year_matrix = normalize(M_temp)
    return normalized_single_year_matrix

def go_rating(player_num = 32, alpha = 1, ndcg_flag = False):
    # 始终启用and 1

    begin_year = 1980
    end_year = 2020
    M_temp = single_matrix(f'{begin_year}-01-01_to_{begin_year + 1}-01-01',player_num=player_num, alpha=alpha)
    flag = 1
    for year in range(begin_year+1, end_year):
        flag += 1
        M_temp += single_matrix(f'{year}-01-01_to_{year+1}-01-01',player_num=player_num, alpha=alpha)
    result_matrix = normalize(M_temp)

    return result_matrix

if __name__ == '__main__':

    os.chdir("..")

    a = go_rating(32,2)

    os.chdir("Matrix_Process")
    df = pd.DataFrame(a)
    df.to_csv("M_result/Go.csv", index=False, header=False)
    # a = single_year()
    # print(a[:4])
    print(a.shape)