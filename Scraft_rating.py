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


def scraft_rating(player_num = 32):
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
    return  result_matrix



if __name__ == '__main__':
    a = scraft_rating()
    print(a)
    print(len(a))