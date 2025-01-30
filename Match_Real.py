"""用真实数据的胜负率来做赛制模拟中的胜负"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr
from sympy import false
from Go_rating import go_rating
from Tennis_rating import tennis_rating
from Badminton_rating import badminton_rating
from Scraft_rating import  scraft_rating
from validation import simulation

def win_judge(win_matrix, p1, p2, player_information):
    """
    通过 胜负率矩阵 判断 p1 是否战胜 p2
    参数：
    - win_matrix: 胜负率矩阵
    - p1, p2: 两个选手的编号

    返回：
    - updated_player_information: 更新后的选手信息 DataFrame
    """
    # 计算 p1 战胜 p2 的胜率，如果存在轮空，则默认选手积分
    if p1 == 'N/A' or p1 is None:
        win_rate = 0
    elif p2 == 'N/A' or p2 is None:
        win_rate = 1
    else:
        win_rate = win_matrix[p1 - 1, p2 - 1]

    # 随机决定是否获胜（根据 win_rate）
    if np.random.rand() < win_rate:
        # p1 战胜 p2，更新 p1 的分数和战胜对手列表
        player_information.loc[player_information['Player'] == p1, 'Score'] += 1
        player_information.loc[player_information['Player'] == p1, 'Defeated_Opponents'].values[0].append(p2)
    else:
        player_information.loc[player_information['Player'] == p2, 'Score'] += 1
        player_information.loc[player_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
    return player_information


def generate_match_pairs(num_rows):
    """
    生成基于轮转思想的对阵表，支持偶数和奇数数量的玩家。
    如果 num_rows 为奇数，则添加一个虚拟的轮空玩家 'N/A'。

    参数：
    - num_rows: 总玩家数量

    返回：
    - match_schedule: 一个包含 num_rows 组轮转对阵表的列表
    """
    is_odd = num_rows % 2 != 0
    if is_odd:
        num_rows += 1  # 添加一个虚拟玩家，使得总人数变为偶数

    # 初始化轮盘，编号从 1 到 num_rows，如果是奇数，则最后一个为 'N/A'
    players = list(range(1, num_rows+1)) if not is_odd else list(range(1, num_rows)) + ['N/A']

    # 保存所有轮次的对阵表
    match_schedule = []
    # 进行 num_rows - 1 轮轮转，每轮生成一组对阵
    for round_num in range(num_rows - 1):
        pairs = []
        for i in range(len(players) // 2):  # 使用 len(players) 代替 num_rows
            pairs.append((players[i], players[len(players) - 1 - i]))
        match_schedule.append(pairs)

        # 轮转操作：保持第一个元素不变，其余元素轮转
        players = [players[0]] + players[1:][1:] + [players[1]]
    return match_schedule

def one_round_match(win_matrix, player_information, schedule):
    for p1, p2 in schedule:
        player_information = win_judge(win_matrix, p1, p2, player_information)
    return player_information

def calculate_opponent_score(player_information):
    """
    计算每个选手的对手小分，即所有战胜过的对手的分数之和。
    """
    opponent_scores = []
    for i, row in player_information.iterrows():
        defeated_opponents = row['Defeated_Opponents']
        opponent_score = player_information.loc[player_information['Player'].isin(defeated_opponents), 'Score'].sum()
        opponent_scores.append(opponent_score)
    player_information['Opponent_Score'] = opponent_scores
    return player_information

def rank_players(player_information):
    """
    对选手进行排名，按总分 -> 对手小分 -> 直接胜负关系排序。
    """
    # 计算对手小分
    player_information = calculate_opponent_score(player_information)

    # 按总分和对手小分排序
    player_information = player_information.sort_values(by=['Score', 'Opponent_Score'], ascending=False).reset_index(drop=True)

    # 处理总分和对手小分都相同的情况：比较直接胜负关系
    ranked_list = [player_information.iloc[0]]  # 添加第一个选手
    for i in range(1, len(player_information)):
        current_player = player_information.iloc[i]
        previous_player = ranked_list[-1]

        # 检查是否需要比较直接胜负关系
        if current_player['Score'] == previous_player['Score'] and current_player['Opponent_Score'] == previous_player['Opponent_Score']:
            # 检查直接胜负关系：当前选手是否战胜了前一个选手
            if current_player['Player'] in previous_player['Defeated_Opponents']:
                ranked_list.insert(len(ranked_list) - 1, current_player)  # 插入到前一个选手之前
            else:
                ranked_list.append(current_player)
        else:
            ranked_list.append(current_player)

    # 转换为 DataFrame
    ranked_df = pd.DataFrame(ranked_list)
    return ranked_df

def robin_round(win_matrix, player_information):
    rr_information = player_information.copy()
    num_rows = rr_information.shape[0]

    match_schedule = generate_match_pairs(num_rows)
    for i in range(num_rows-1):
        rr_information = one_round_match(win_matrix, rr_information, match_schedule[i])

    ranked_rr_information = rank_players(rr_information)

    return rr_information, ranked_rr_information

def sr_first_round_match_list(player_information):
    # 在瑞士轮中，第一轮乱序排列
    match_list = []
    players = player_information["Player"].tolist()
    np.random.shuffle(players)
    while players:
        p1 = players.pop(0)
        p2 = players.pop(0) if players else None
        match_list.append((p1, p2))
    return match_list

def sr_one_round_match_list(player_information):
    
    match_list = []

    # 按得分和实力降序排列
    player_information = player_information.sort_values(by=["Score"], ascending=[False])
    players = player_information["Player"].tolist()
    scores = player_information["Score"].tolist()
    defeated_opponents = player_information["Defeated_Opponents"].tolist()
    used_players = set() # 记录已匹配过的选手

    while players:
        p1 = players.pop(0)  # 选出当前最高分的选手
        if p1 in used_players:
            continue  # 如果该选手已匹配过，则跳过

        # 尝试为 p1 找到一个合适的对手 p2
        for i, p2 in enumerate(players):
            if p2 not in defeated_opponents[p1 - 1]:  # 检查 p1 和 p2 是否对战过
                match_list.append((p1, p2))  # 匹配成功
                used_players.add(p1)
                used_players.add(p2)
                players.pop(i)  # 将 p2 移出待匹配列表
                break

    if len(players) == 1:
        p1 = players[0]
        match_list.append((p1, None))  # None 表示轮空

    return match_list


def swiss_round(win_matrix, player_information, round_num):
    sr_information = player_information.copy()
    num_rows = sr_information.shape[0]
    for i in range(round_num):
        if i == 0:
            one_round_list = sr_first_round_match_list(player_information)
        else:
            one_round_list = sr_one_round_match_list(player_information)
        sr_information = one_round_match(win_matrix, sr_information, one_round_list)
    ranked_sr_information = rank_players(sr_information)
    return sr_information, ranked_sr_information

def calculate_coefficient(player_information, ranked_information):
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

def calculate_ndcg_coefficient(player_information, ranked_information):
    """
    计算引入 NDCG 思想的 Spearman 相关系数，衡量初始实力排名与比赛结果排名的相关性。
    
    参数：
    - player_information: 初始选手信息的 DataFrame，按实力排序，包含 'player_id' 和 'initial_rank' 列
    - ranked_information: 比赛结果排名后的 DataFrame，按得分排名，包含 'player_id' 和 'final_rank' 列

    返回：
    - ndcg_spearman_score: 引入 NDCG 思想的 Spearman 相关系数
    """
    # 获取初始排名和比赛结果排名
    initial_rank = player_information['Player'].values
    final_rank = ranked_information['Player'].values
    
    # 计算折扣因子
    n = len(initial_rank)
    discount_factors = 1 / np.log2(np.arange(1, n + 1) + 1)  # 折扣因子
    
    # 计算加权的排名相关性
    weighted_initial_rank = initial_rank * discount_factors
    weighted_final_rank = final_rank * discount_factors
    
    # 计算 Spearman 相关系数
    correlation, _ = spearmanr(weighted_initial_rank, weighted_final_rank)
    
    return correlation

def standard_matrix(rows=3, cols=3):
    """
    Generates a matrix with the given dimensions, filled with the value 0.5.

    Parameters:
        rows (int): Number of rows in the matrix (default is 3).
        cols (int): Number of columns in the matrix (default is 3).

    Returns:
        numpy.ndarray: A matrix filled with 0.5.
    """
    return np.full((rows, cols), 0.5)

def save_data(data, filename):
    # 如果data是列表，则将其转换为DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data)
    data.to_csv(filename, index=False)

def plot_result(scores_result, result_path):
    """
    绘制scores_result结果的柱状图，其中不同类型的比赛要分隔开，每种比赛包括四个柱子，
    纵轴表示Spearman相关系数的大小。
    """
    rcParams['font.sans-serif'] = ['Times New Roman'] 
    rcParams['axes.unicode_minus'] = False
    match_types = scores_result['Match'].unique()
    bar_width = 0.2
    index = np.arange(len(match_types))
    spearman_columns = ['Spearman_RR', 'Spearman_SR', 'NDCG_Spearman_RR', 'NDCG_Spearman_SR']
    
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(spearman_columns):
        values = [scores_result[scores_result['Match'] == match][col].mean() for match in match_types]
        plt.bar(index + i * bar_width, values, bar_width, label=col)
    
    plt.xlabel('Match Type')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient of different match types')
    plt.xticks(index + bar_width * (len(spearman_columns) - 1) / 2, match_types)
    plt.legend(title="Coefficient Type")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "coefficient_of_different_match_types.png"))
    plt.show()

if __name__ == "__main__":
    result_path = 'MatchRealResult'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    match_set = ['go','tennis','badminton','scraft']

    # 保存所有NDCG Spearman和Spearman相关系数,以及不同比赛类型的结果
    scores_result = pd.DataFrame({
        "Match": match_set,
        "Spearman_RR": [0] * len(match_set),
        "Spearman_SR": [0] * len(match_set),
        "NDCG_Spearman_RR": [0] * len(match_set),
        "NDCG_Spearman_SR": [0] * len(match_set)
    })

    # 如果只保留前32个选手，则rating_num_32=True
    rating_num_32 = False

    #胜负矩阵
    winning_matrix = {
    match: globals().get(f"{match}_rating", lambda x: None)(rating_num_32)
    for match in match_set
    }

    winning_matrix['Random'] = standard_matrix(rows=33, cols=33)

    match_to_matrix = {
    'go': winning_matrix['go'],
    'tennis': winning_matrix['tennis'],
    'badminton': winning_matrix['badminton'],
    'scraft': winning_matrix['scraft'],
    # 'random': winning_matrix['Random']
    }

    for match in match_set:
        # 读取胜率矩阵，并生成一个初始选手信息的DataFrame
        win_matrix = winning_matrix[match]
        Initial_Player_DataFrame = pd.DataFrame({
            "Player": range(1, win_matrix.shape[0] + 1),  # Ensure the correct length for the "Player" column
            "Score": [0] * win_matrix.shape[0],  # Ensure "Score" column is the same length
            "Defeated_Opponents": [[] for _ in range(win_matrix.shape[0])]  # Create a list of empty lists, one for each player
        })

        # 求多次实验下的均值，增强稳定性
        spearman_score_rr_time = []
        ndcg_spearman_score_rr_time= []
        spearman_score_sr_time = []
        ndcg_spearman_score_sr_time = []
        for i in range(1):
            rr, ranked_rr = robin_round(win_matrix, Initial_Player_DataFrame)
            sr, ranked_sr = swiss_round(win_matrix, Initial_Player_DataFrame, round_num=30)

            spearman_score_rr = calculate_coefficient(rr,ranked_rr)
            ndcg_spearman_score_rr = calculate_ndcg_coefficient(rr,ranked_rr)

            spearman_score_sr = calculate_coefficient(sr,ranked_sr)
            ndcg_spearman_score_sr = calculate_ndcg_coefficient(sr,ranked_sr)

            spearman_score_rr_time.append(spearman_score_rr)
            ndcg_spearman_score_rr_time.append(ndcg_spearman_score_rr)
            spearman_score_sr_time.append(spearman_score_sr)
            ndcg_spearman_score_sr_time.append(ndcg_spearman_score_sr)
        
        # 保存多次运行下的平均结果
        scores_result.loc[scores_result['Match'] == match, 'Spearman_RR'] = np.mean(spearman_score_rr_time)
        scores_result.loc[scores_result['Match'] == match, 'Spearman_SR'] = np.mean(spearman_score_sr_time)
        scores_result.loc[scores_result['Match'] == match, 'NDCG_Spearman_RR'] = np.mean(ndcg_spearman_score_rr_time)
        scores_result.loc[scores_result['Match'] == match, 'NDCG_Spearman_SR'] = np.mean(ndcg_spearman_score_sr_time)

        # save_data(spearman_score_rrs, os.path.join(result_path, f"{match}_spearman_score_rrs.txt"))
        # save_data(ndcg_spearman_score_rrs, os.path.join(result_path, f"{match}_ndcg_spearman_score_rrs.txt"))
        # save_data(spearman_score_srs, os.path.join(result_path, f"{match}_spearman_score_srs.txt"))
        # save_data(ndcg_spearman_score_srs, os.path.join(result_path, f"{match}_ndcg_spearman_score_srs.txt"))
    # 绘制scores_result结果的柱状图，其中不同类型的比赛要分隔开，每种比赛包括四个柱子，分别是Spearman_RR、Spearman_SR、NDCG_Spearman_RR、NDCG_Spearman_SR
    plot_result(scores_result, result_path)
        