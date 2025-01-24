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


def single_matrix(filename, player_num = 32):
    rank_list_temp_ = read_rankings(f'Data/Go_64_data/{filename}/player_rankings.txt')
    rank_list_temp = rank_list_temp_[:player_num]
    win_matrix = np.zeros((player_num+1, player_num+1), dtype=int)

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
                        win_matrix[player_idx, player_num] += 1
                    elif result == "负":
                        win_matrix[player_num, player_idx] += 1
                elif opponent_idx is not None:
                    if result == "负":
                        win_matrix[opponent_idx, player_num] += 1
                    elif result == "胜":
                        win_matrix[player_num, opponent_idx] += 1
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

def convert_to_33x33_ndcg(matrix, player_num=32):
    # 保留前 player_num x player_num 的部分
    top_32 = matrix[:player_num, :player_num]

    # 计算第 i 行在第归并列之后的损失因子加权平均
    avg_col_last_modified = np.array([
        np.sum([
            matrix[i, j] / (np.log2(j - player_num + 2) + 1)
            for j in range(player_num, matrix.shape[1])
            if matrix[i, j] > 0 or matrix[j, i] > 0
        ]) / np.sum([
            1 / (np.log2(j - player_num + 2) + 1)
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
        ]) / np.sum([
            1 / (np.log2(i - player_num + 2) + 1)
            for i in range(player_num, matrix.shape[0])
            if matrix[i, j] > 0 or matrix[j, i] > 0
        ]) if np.any([
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

def single_year(player_num=32):
    begin_year = 1980
    M_temp = single_matrix(f'{begin_year}-01-01_to_{begin_year + 1}-01-01',player_num=player_num)
    normalized_matrix = normalize(M_temp)
    return normalized_matrix

def go_rating(rating_num_32, player_num = 32):
    begin_year = 1980
    end_year = 2020
    M_temp = single_matrix(f'{begin_year}-01-01_to_{begin_year + 1}-01-01',player_num=player_num)
    flag = 1
    for year in range(begin_year+1, end_year):
        flag += 1
        M_temp += single_matrix(f'{year}-01-01_to_{year+1}-01-01')
    result_matrix = normalize(M_temp)
    if rating_num_32 == False:
        result_matrix = convert_to_33x33_ndcg(result_matrix)
    return result_matrix

if __name__ == '__main__':
    a = go_rating()
    # a = single_year()
    print(a[:4])