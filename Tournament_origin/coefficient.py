import numpy as np
import pandas as pd
import numpy as np

###########################################################
####################### 相关系数计算 ########################
###########################################################
def calculate_Spearman_coefficient(player_information, ranked_information):
    """
    计算 Spearman 相关系数，衡量初始实力排名与比赛结果排名的相关性。

    参数：
    - player_information: 初始选手信息的 DataFrame，按实力排序
    - ranked_information: 比赛结果排名后的 DataFrame，按得分排名

    返回：
    - spearman_score: 计算得到的 Spearman 相关系数
    """
    # 提取选手编号列表
    initial_players = player_information['Player'].tolist()
    ranked_players = ranked_information['Player'].tolist()

    # 将初始排名和比赛排名转换为 Pandas Series，索引为 Player
    initial_rank = pd.Series(range(1, len(initial_players) + 1), index=initial_players)
    result_rank = pd.Series(range(1, len(ranked_players) + 1), index=ranked_players)

    # 计算排名差的平方和
    d_squared_sum = np.sum((initial_rank - result_rank) ** 2)

    n = len(player_information)

    # 计算 Spearman 相关系数
    spearman_score = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

    return spearman_score

def calculate_ndcg_Spearman_coefficient(player_information, ranked_information):
    """
    计算 NDCG-Spearman 相关系数，结合了排名加权的思想，排名越靠前的选手越重要。
    通过给排名乘上对数衰减因子来增强排名靠前选手的重要性。
    """
    initial_rank = player_information['Player'].rank()
    result_rank = ranked_information['Player'].rank()

    x_bar = initial_rank.mean()
    y_bar = result_rank.mean()

    # 计算分子：∑ (xi - x̄) * (yi - ȳ) / log2(i+1)
    numerator = np.sum(((initial_rank - x_bar) * (result_rank - y_bar)) / np.log2(np.arange(1, len(initial_rank) + 1) + 1))

    # 计算分母：√(∑(xi - x̄)² / log2(i+1) * ∑(yi - ȳ)² / log2(i+1))
    denominator = np.sqrt(np.sum(((initial_rank - x_bar) ** 2) / np.log2(np.arange(1, len(initial_rank) + 1) + 1)) *
                          np.sum(((result_rank - y_bar) ** 2) / np.log2(np.arange(1, len(result_rank) + 1) + 1)))

    ndcg_spearman_score = numerator / denominator

    return ndcg_spearman_score

