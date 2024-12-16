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


def single_matrix(filename):
    rank_list_temp = read_rankings(f'Go_data/{filename}/player_rankings.txt')

    win_matrix = np.zeros((33, 33), dtype=int)

    input_file = f'Data/Go_data/{filename}/player_games.txt'  # 替换为你的文件路径
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
                        win_matrix[player_idx, 32] += 1
                    elif result == "负":
                        win_matrix[32, player_idx] += 1
                elif opponent_idx is not None:
                    if result == "负":
                        win_matrix[opponent_idx, 32] += 1
                    elif result == "胜":
                        win_matrix[32, opponent_idx] += 1
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

def go_rating():
    begin_year = 1980
    end_year = 2020
    M_temp = single_matrix(f'{begin_year}-01-01_to_{begin_year + 1}-01-01')
    flag = 1
    for year in range(begin_year+1, end_year):
        flag +=1
        M_temp += single_matrix(f'{year}-01-01_to_{year+1}-01-01')
    normalized_matrix = normalize(M_temp)
    # print(normalized_matrix)
    return normalized_matrix

if __name__ == '__main__':
    go_rating()