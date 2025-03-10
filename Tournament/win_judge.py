import numpy as np
import pandas as pd
import random
import numpy as np

###########################################################
################## 输赢机制，破分机制 ########################
###########################################################
def generate_random_match_pairs(players):
    """
    随机配对选手进行比赛，不修改原始选手列表。
    """
    players_copy = players.copy()  # 复制一份，避免修改原列表
    random.shuffle(players_copy)
    match_schedule = []
    
    while len(players_copy) > 1:
        p1 = players_copy.pop(0)
        p2 = players_copy.pop(0)
        match_schedule.append((p1, p2))
    
    # 处理轮空选手
    if len(players_copy) == 1:
        match_schedule.append((players_copy.pop(), 'N/A'))
    
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

def win_judge_with_lose_time(win_matrix, p1, p2, player_information):
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
        player_information.loc[player_information['Player'] == p2, 'Lose_Time'] += 1
        return player_information
    else:
        if p2 != 'N/A' and p2 is not None:
            player_information.loc[player_information['Player'] == p2, 'Score'] += 1
            player_information.loc[player_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
            player_information.loc[player_information['Player'] == p1, 'Lose_Time'] += 1
            return player_information

def win_judge_with_weight(win_matrix, p1, p2, weight, player_information):
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

    # 如果是第一轮，则设置为1
    weight = 1 if weight == 0 else weight
    # 随机决定是否获胜（根据 win_rate）
    if np.random.rand() < win_rate:
        # p1 战胜 p2，更新 p1 的分数和战胜对手列表
        player_information.loc[player_information['Player'] == p1, 'Score'] += weight
        player_information.loc[player_information['Player'] == p1, 'Defeated_Opponents'].values[0].append(p2)
        return player_information
    else:
        if p2 != 'N/A' and p2 is not None:
            player_information.loc[player_information['Player'] == p2, 'Score'] += weight
            player_information.loc[player_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
            return player_information


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
    对选手进行排名，按总分 -> 对手小分 -> 直接胜负关系排序 -> tie-breaker排序。
    当总分和对手小分相同时，如果直接胜负关系无法区分，则使用随机生成的 tie-breaker 值进行排序，
    避免默认使用选手编号排序。
    """
    # 计算对手小分
    player_information = calculate_opponent_score(player_information)
    
    # 为每个选手增加一个随机 tie-breaker 值
    player_information = player_information.copy()
    player_information["TieBreaker"] = np.random.rand(len(player_information))
    
    # 按总分、对手小分、tie-breaker排序（均降序排序）
    player_information = player_information.sort_values(
        by=['Score', 'Opponent_Score', 'TieBreaker'], 
        ascending=False
    ).reset_index(drop=True)
    
    # 针对总分和对手小分相同的情况，使用直接胜负关系重新调整顺序
    ranked_list = [player_information.iloc[0]]  # 添加第一位选手
    for i in range(1, len(player_information)):
        current_player = player_information.iloc[i]
        previous_player = ranked_list[-1]
        
        # 如果当前选手与前一位选手总分和对手小分相同，则检查直接胜负关系
        if (current_player['Score'] == previous_player['Score'] and 
            current_player['Opponent_Score'] == previous_player['Opponent_Score']):
            # 若当前选手曾战胜前一位选手，则应排在前面
            if current_player['Player'] in previous_player['Defeated_Opponents']:
                ranked_list.insert(len(ranked_list) - 1, current_player)
            # 若前一位选手曾战胜当前选手，则当前选手放后面
            elif previous_player['Player'] in current_player['Defeated_Opponents']:
                ranked_list.append(current_player)
            else:
                # 如果双方没有直接对阵，则依赖 tie-breaker 保持原排序
                ranked_list.append(current_player)
        else:
            ranked_list.append(current_player)
    
    ranked_df = pd.DataFrame(ranked_list)
    return ranked_df
