"""用真实数据的胜负率来做赛制模拟中的胜负"""
import numpy as np
import pandas as pd
import random
import numpy as np
import os
from matplotlib import rcParams
from .coefficient import calculate_Spearman_coefficient
from .coefficient import calculate_ndcg_Spearman_coefficient
from .tounaments import robin_round, swiss_round, double_elimination_random, weighted_round_robin, rr_knockout, ladder_tournament

def standard_matrix(rows, cols):
    """
    Generates a matrix with the given dimensions, filled with the value 0.5.

    Parameters:
        rows (int): Number of rows in the matrix (default is 3).
        cols (int): Number of columns in the matrix (default is 3).

    Returns:
        numpy.ndarray: A matrix filled with 0.5.
    """
    return np.full((rows, cols), 0.5)

# # 方便每次初始化
class PlayerData:
    def __init__(self, player_count):
        self.df = pd.DataFrame({
            "Player": range(1, player_count + 1),
            "Score": [0.0] * player_count,
            "Defeated_Opponents": [[] for _ in range(player_count)]
        })
    def get_copy(self):
        return self.df.copy(deep=True)

###########################################################
################## 调用这个函数来运行赛制模拟 #################
###########################################################
def tournament_correlation(match_name, distribution_type, win_matrix, iterations, datatype):   
    Initial_Player_DataFrame = pd.DataFrame({
        "Player": range(1, win_matrix.shape[0] + 1),
        "Score": [0.0] * win_matrix.shape[0],
        "Defeated_Opponents": [[] for _ in range(win_matrix.shape[0])]
    })

    spearman_score_rr = 0
    spearman_score_sr = 0
    spearman_score_de = 0
    spearman_score_weighted = 0
    spearman_score_rr_knockout = 0
    spearman_score_ladder = 0

    ndcg_spearman_score_rr = 0
    ndcg_spearman_score_sr = 0
    ndcg_spearman_score_de = 0
    ndcg_spearman_score_weighted = 0
    ndcg_spearman_score_rr_knockout = 0
    ndcg_spearman_score_ladder = 0

    player_data = PlayerData(win_matrix.shape[0])
    for i in range(iterations):
        # 循环赛
        rr_initial, rr_ranked = robin_round(win_matrix, player_data.get_copy())

        # 瑞士轮（假设进行30轮）
        # 这个地方有个bug，如果每次不重新创建Initial_Player_DataFrame的话，结果会叠加，有点不太清楚应该怎么消除这个bug
        player_data = PlayerData(win_matrix.shape[0])
        sr_initial, sr_ranked = swiss_round(win_matrix, player_data.get_copy(), round_num=30)

        # 随机双淘汰赛
        player_data = PlayerData(win_matrix.shape[0])
        de_initial, de_ranked = double_elimination_random(win_matrix, player_data.get_copy())

        # 加权循环赛
        player_data = PlayerData(win_matrix.shape[0])
        weighted_initial, weighted_ranked = weighted_round_robin(win_matrix, player_data.get_copy())

        # 分组赛+淘汰赛
        player_data = PlayerData(win_matrix.shape[0])
        rr_knockout_initial, rr_knockout_ranked = rr_knockout(win_matrix, player_data.get_copy())

        # 阶梯赛
        player_data = PlayerData(win_matrix.shape[0])
        ladder_initial, ladder_ranked = ladder_tournament(win_matrix, player_data.get_copy())

        spearman_score_rr += calculate_Spearman_coefficient(rr_initial, rr_ranked)
        spearman_score_sr += calculate_Spearman_coefficient(sr_initial, sr_ranked)
        spearman_score_de += calculate_Spearman_coefficient(de_initial, de_ranked)
        spearman_score_weighted += calculate_Spearman_coefficient(weighted_initial, weighted_ranked)
        spearman_score_rr_knockout += calculate_Spearman_coefficient(rr_knockout_initial, rr_knockout_ranked)
        spearman_score_ladder += calculate_Spearman_coefficient(ladder_initial, ladder_ranked)

        ndcg_spearman_score_rr += calculate_ndcg_Spearman_coefficient(rr_initial, rr_ranked)
        ndcg_spearman_score_sr += calculate_ndcg_Spearman_coefficient(sr_initial, sr_ranked)          
        ndcg_spearman_score_de += calculate_ndcg_Spearman_coefficient(de_initial, de_ranked)
        ndcg_spearman_score_weighted += calculate_ndcg_Spearman_coefficient(weighted_initial, weighted_ranked)
        ndcg_spearman_score_rr_knockout += calculate_ndcg_Spearman_coefficient(rr_knockout_initial, rr_knockout_ranked)
        ndcg_spearman_score_ladder += calculate_ndcg_Spearman_coefficient(ladder_initial, ladder_ranked)
    
    # 保存结果
    scores_result = pd.DataFrame({
        "Match": [match_name],
        "Distribution Type": [distribution_type],
        "Data Type": [datatype],
        # Spearman
        "Spearman Round Robin": [spearman_score_rr/iterations],
        "Spearman Swiss Round": [spearman_score_sr/iterations],
        "Spearman Double Elimination": [spearman_score_de/iterations],
        "Spearman Weighted Round Robin": [spearman_score_weighted/iterations],
        "Spearman Round Robin Knockout": [spearman_score_rr_knockout/iterations],
        "Spearman Ladder Tournament": [spearman_score_ladder/iterations],
        # # NDCG_Spearman
        "NDCG_Spearman Round Robin": [ndcg_spearman_score_rr/iterations],
        "NDCG_Spearman Swiss Round": [ndcg_spearman_score_sr/iterations],
        "NDCG_Spearman Double Elimination": [ndcg_spearman_score_de/iterations],
        "NDCG_Spearman Weighted Round Robin": [ndcg_spearman_score_weighted/iterations],
        "NDCG_Spearman Round Robin Knockout": [ndcg_spearman_score_rr_knockout/iterations],
        "NDCG_Spearman Ladder Tournament": [ndcg_spearman_score_ladder/iterations]
    })
    return scores_result
        
if __name__=='__main__':
    tournament_correlation("Go", "Normal", standard_matrix(32, 32), 1, 'Real Data')