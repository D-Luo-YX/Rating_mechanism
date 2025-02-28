import numpy as np
import pandas as pd
import random
import numpy as np

###########################################################
################## 输赢机制，破分机制 ########################
###########################################################
def generate_random_match_pairs(players):
    """
    随机配对选手进行比赛
    """
    random.shuffle(players)
    match_schedule = []
    while len(players) > 1:
        p1 = players.pop(0)
        p2 = players.pop(0)
        match_schedule.append((p1, p2))

    # 轮空选手
    if len(players) == 1:
        match_schedule.append((players.pop(), 'N/A'))

    return match_schedule


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


def win_judge_return_winner(win_matrix, p1, p2, player_information):
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
        return p1
    else:
        if p2 != 'N/A' and p2 is not None:
            player_information.loc[player_information['Player'] == p2, 'Score'] += 1
            player_information.loc[player_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
            return p2

def one_round_match(win_matrix, player_information, schedule):
    """
    进行一轮比赛，更新选手信息 DataFrame
    """
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
