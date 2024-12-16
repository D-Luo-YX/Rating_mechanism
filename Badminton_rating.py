import numpy as np
import pandas as pd
import csv
import os


def process_match_file(file_path):
    """
    处理比赛记录文件，生成33*33的矩阵。
    file_path: str, 输入文件路径
    return: np.ndarray, 33*33矩阵
    """
    win_matrix = np.zeros((33, 33), dtype=int)

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("date"):
                continue

            try:
                parts = line.strip().split(",")
                _, player1, rank1, result, player2, rank2 = parts

                rank1 = int(rank1) if rank1.isdigit() and int(rank1) <= 33 else 33
                rank2 = int(rank2) if rank2.isdigit() and int(rank2) <= 33 else 33

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

def badminton_rating():
    input_file = 'Data/Badminton_data/badminton_games.txt'
    output_file = 'Data/Badminton_data/matrix.txt'
    temp_matrix = process_match_file(input_file)
    winning_matrix = normalize(temp_matrix)
    return winning_matrix

if __name__ == "__main__":
    a = badminton_rating()
    print(a)