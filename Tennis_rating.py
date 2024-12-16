import numpy as np
import pandas as pd
import csv
import os


### Converting the .txt to the Matrix
def process_match_file(input_file, output_file):
    matrix = np.zeros((33, 33), dtype=int)

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

            rank_a = min(player_a_rank, 33)
            rank_b = min(player_b_rank, 33)

            if player_a_rank > 32 and player_b_rank > 32:
                continue

            if player_a_score > player_b_score:
                matrix[rank_a - 1][rank_b - 1] += 1
            elif player_b_score > player_a_score:
                matrix[rank_b - 1][rank_a - 1] += 1

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
    return normalized_M

def tennis_rating():
    input_file = 'Data/Tennis_data/match_data.txt'
    output_file = 'Data/Tennis_data/matrix.txt'
    temp_matrix = process_match_file(input_file, output_file)
    winning_matrix = normalize(temp_matrix)
    return winning_matrix

if __name__ == "__main__":
    a = tennis_rating()
    print(a)