"""
Tournament Module
-----------------
本模块提供以下比赛赛制的对阵生成及结果处理函数：
  1. 循环赛（Round Robin）
  2. 瑞士轮（Swiss System）
  3. 双淘汰赛（Double Elimination）
  4. 加权循环赛（Weighted Round Robin）
  5. 分组赛+淘汰赛（Group Stage + Knockout Stage）
  6. 阶梯赛（Ladder Tournament）

各函数的参数要求说明：
  - win_matrix: 胜负判断矩阵（如 NumPy 数组或 DataFrame），用于判断比赛结果。
  - player_information: 包含选手信息的 DataFrame，必须包含 'Player' 列，其它列（如 'Score'、'Defeated_Opponents'、'Lose_Time'）根据赛制要求。
  - rounds_num / matches_num: 控制比赛的轮次数或总场次数。要求其中一个设置为 None，表示采用另一种控制方式（finish_all_rounds 为 False 时）。
  - finish_all_rounds: 布尔值，若为 True，则忽略 rounds_num 与 matches_num，采用预设规则（例如循环赛完成所有轮次，或阶梯赛仅进行一轮）。

外部依赖函数包括：
  - win_judge_return_winner: 根据 win_matrix 判断比赛胜者，返回胜者编号。
  - win_judge_with_weight: 考虑权重的胜负判断函数，用于加权循环赛。
  - win_judge_with_lose_time: 用于双淘汰赛，更新选手输的次数等信息。
  - generate_random_match_pairs: 随机生成淘汰赛对阵表（已调整为不修改原选手列表）。
  - one_round_match: 执行一轮比赛，更新选手信息（循环赛、瑞士轮使用）。
  - rank_players: 对选手进行排名排序。
"""

import math
import random
import numpy as np
import pandas as pd

from .win_judge import (
    win_judge_return_winner, 
    win_judge_with_weight, 
    generate_random_match_pairs, 
    one_round_match, 
    rank_players, 
    win_judge_with_lose_time
)
# from win_judge import (
#     win_judge_return_winner, 
#     win_judge_with_weight, 
#     generate_random_match_pairs, 
#     one_round_match, 
#     rank_players, 
#     win_judge_with_lose_time
# )

########################################################################
# 循环赛（Round Robin）相关函数
########################################################################

def generate_match_pairs(num_rows):
    """
    生成基于轮转思想的对阵表，适用于循环赛。
    
    说明：
      - 每个选手与其他选手对阵一次。
      - 若总玩家数为奇数，则添加一个虚拟玩家 'N/A' 以保证轮转正常。
    
    参数：
      - num_rows: 整数，总玩家数量（不含虚拟玩家）。
    
    返回：
      - match_schedule: 包含 (num_rows - 1) 轮对阵表的列表，每一轮为一组对阵，
                        对阵以元组 (选手1, 选手2) 表示。
    """
    is_odd = num_rows % 2 != 0
    if is_odd:
        num_rows += 1  # 增加虚拟玩家

    # 初始化选手列表：若原始为奇数，则最后一个为 'N/A'
    players = list(range(1, num_rows+1)) if not is_odd else list(range(1, num_rows)) + ['N/A']
    # random.shuffle(players)
    match_schedule = []

    for _ in range(num_rows - 1):
        pairs = []
        for i in range(len(players) // 2):
            pairs.append((players[i], players[len(players) - 1 - i]))
        match_schedule.append(pairs)
        # 轮转：保持第一个选手不变，其余选手轮转
        players = [players[0]] + players[1:][1:] + [players[1]]
    return match_schedule

def robin_round(win_matrix, player_information, rounds_num, matches_num, finish_all_rounds):
    """
    循环赛比赛：每个选手与其他选手对阵一次。
    
    参数：
      - win_matrix: 胜负判断矩阵，用于确定每场比赛胜者。
      - player_information: DataFrame，包含选手信息，必须含 'Player' 列。
      - rounds_num: 整数，指定循环赛阶段的比赛轮数；若采用场次数控制，则设置为 None。
      - matches_num: 整数，指定循环赛阶段的总比赛场次数；若采用轮次数控制，则设置为 None。
      - finish_all_rounds: 布尔值，若为 True，则执行所有轮次，不受 rounds_num 或 matches_num 限制。
    
    返回：
      - 原始 player_information 与经过循环赛后的排序信息 DataFrame（由 rank_players 得到）。
    """
    rr_information = player_information.copy()
    num_rows = rr_information.shape[0]
    match_schedule = generate_match_pairs(num_rows)
    matches_count = 0

    if finish_all_rounds:
        # 保留所有生成的轮次
        pass
    elif matches_num is None:
        if rounds_num > num_rows:
            # print("轮次数超过最多可能轮次数，自动调整。")
            rounds_num = num_rows
        match_schedule = match_schedule[:rounds_num]
    elif rounds_num is None:
        # if matches_num < math.ceil(num_rows / 2):
        #     print("比赛场次数小于选手数量，无法完成一轮。")
        truncated_schedule = []
        for round_pairs in match_schedule:
            round_truncated = []
            for pair in round_pairs:
                if matches_count < matches_num:
                    round_truncated.append(pair)
                    matches_count += 1
            if round_truncated:
                truncated_schedule.append(round_truncated)
        match_schedule = truncated_schedule

    for round_pairs in match_schedule:
        rr_information = one_round_match(win_matrix, rr_information, round_pairs)
    ranked_rr_information = rank_players(rr_information)
    return player_information, ranked_rr_information

########################################################################
# 瑞士轮（Swiss System）相关函数
########################################################################

def sr_first_round_match_list(player_information, total_matches_played, matches_num):
    """
    瑞士轮第一轮比赛对阵生成：随机乱序配对。
    
    参数：
      - player_information: DataFrame，包含 'Player' 列。
      - total_matches_played: 整数，当前已进行比赛场次（可为 None）。
      - matches_num: 整数，总比赛场次限制（可为 None）。
    
    返回：
      - match_list: 第一轮对阵列表，每个对阵以 (p1, p2) 表示，p2 为 None 表示轮空。
      - total_matches_played: 更新后的比赛场次数。
    """
    match_list = []
    players = player_information["Player"].tolist()
    np.random.shuffle(players)
    while players:
        p1 = players.pop(0)
        p2 = players.pop(0) if players else None
        match_list.append((p1, p2))
        if total_matches_played is not None:
            total_matches_played += 1
        if matches_num is not None and total_matches_played == matches_num:
            break
    return match_list, total_matches_played

def sr_one_round_match_list(player_information, total_matches_played, matches_num):
    """
    瑞士轮后续轮次比赛对阵生成：按得分降序排列后匹配未交手选手。
    
    参数同上。
    
    返回：
      - match_list: 当前轮次的对阵列表。
      - total_matches_played: 更新后的比赛场次数。
    """
    match_list = []
    player_information = player_information.sort_values(by=["Score"], ascending=False)
    players = player_information["Player"].tolist()
    used_players = set()
    while players:
        p1 = players.pop(0)
        if p1 in used_players:
            continue
        for i, p2 in enumerate(players):
            if p2 not in player_information[player_information["Player"] == p1]["Defeated_Opponents"].iloc[0] and p1 not in player_information[player_information["Player"] == p2]["Defeated_Opponents"].iloc[0]:
                match_list.append((p1, p2))
                used_players.add(p1)
                used_players.add(p2)
                players.pop(i)
                break
        if total_matches_played is not None:
            total_matches_played += 1
        if matches_num is not None and total_matches_played == matches_num:
            break
        if len(players) == 1:
            p1 = players[0]
            match_list.append((p1, None))
    return match_list, total_matches_played

def swiss_round(win_matrix, player_information, round_num, matches_num, finish_all_rounds):
    """
    瑞士轮比赛：支持按轮次数或总场次数控制。
    
    参数：
      - win_matrix, player_information 同上。
      - round_num: 指定比赛轮数；若采用场次数控制，则为 None。
      - matches_num: 指定比赛场次数；若采用轮次数控制，则为 None。
      - finish_all_rounds: 若为 True，则按选手数量计算轮数（ceil(log2(num_rows)））。
    
    返回：
      - 原始 player_information 与瑞士轮比赛后的排名结果 DataFrame。
    """
    sr_information = player_information.copy()
    num_rows = sr_information.shape[0]
    total_matches_played = 0

    if finish_all_rounds:
        round_num = math.ceil(math.log2(num_rows))
        matches_num = None

    if matches_num is None:
        for i in range(round_num):
            if i == 0:
                one_round_list, _ = sr_first_round_match_list(sr_information, None, None)
            else:
                one_round_list, _ = sr_one_round_match_list(sr_information, None, None)
            # 如果是最后一轮，不需要更新选手信息
            if len(one_round_list) < math.ceil(num_rows/2):
                break
            sr_information = one_round_match(win_matrix, sr_information, one_round_list)    
        ranked_sr_information = rank_players(sr_information)
        return player_information, ranked_sr_information
    elif round_num is None:
        for i in range(1000):
            if i == 0:
                one_round_list, total_matches_played = sr_first_round_match_list(sr_information, total_matches_played, matches_num)
            else:
                one_round_list, total_matches_played = sr_one_round_match_list(sr_information, total_matches_played, matches_num)
            sr_information = one_round_match(win_matrix, sr_information, one_round_list)
            if total_matches_played == matches_num or len(one_round_list) < math.ceil(num_rows/2):
                break
        ranked_sr_information = rank_players(sr_information)
        return player_information, ranked_sr_information

########################################################################
# 双淘汰赛（Double Elimination）相关函数
########################################################################

def generate_double_elimination_pairs(player_information, is_first_time, total_matches_num, matches_num):
    """
    生成双淘汰赛对阵表：
      - 第一轮随机配对；
      - 后续轮次根据选手输的次数进行配对，若选手数为奇数，安排轮空。
    
    参数：
      - player_information: DataFrame，包含 'Player' 及 'Lose_Time' 列。
      - is_first_time: 布尔值，表示是否为第一轮。
      - total_matches_num: 当前累计比赛场数。
      - matches_num: 总比赛场数限制（可为 None）。
    
    返回：
      - match_schedule: 对阵列表，每场以 (p1, p2) 表示。
      - total_matches_num: 更新后的比赛场次数。
    """
    if is_first_time:
        players = player_information["Player"].tolist()
        random.shuffle(players)
        match_schedule = []
        while len(players) > 1:
            p1 = players.pop(0)
            p2 = players.pop(0)
            match_schedule.append((p1, p2))
            total_matches_num += 1 
            if matches_num is not None and total_matches_num == matches_num:
                break
        if len(players) == 1:
            match_schedule.append((players.pop(), 'N/A'))
        return match_schedule, total_matches_num
    else:
        # 最后一场比赛
        if len(player_information[player_information['Lose_Time'] == 1]["Player"].tolist()) == 1 and \
           len(player_information[player_information['Lose_Time'] == 0]["Player"].tolist()) == 1:
            match_schedule = []
            p1 = player_information[player_information['Lose_Time'] == 0]["Player"].tolist()[0]
            p2 = player_information[player_information['Lose_Time'] == 1]["Player"].tolist()[0]
            match_schedule.append((p1, p2))
            total_matches_num += 1 
            return match_schedule, total_matches_num

        match_schedule = []
        players_no_lose = player_information[player_information['Lose_Time'] == 0]["Player"].tolist()
        while len(players_no_lose) > 1:
            p1 = players_no_lose.pop(0)
            p2 = players_no_lose.pop(0)
            match_schedule.append((p1, p2))
            total_matches_num += 1 
            if matches_num is not None and total_matches_num == matches_num:
                break
        if len(players_no_lose) == 1:
            match_schedule.append((players_no_lose.pop(), 'N/A'))
        
        players_one_lose = player_information[player_information['Lose_Time'] == 1]["Player"].tolist()
        while len(players_one_lose) > 1:
            if matches_num is not None and total_matches_num == matches_num:
                break
            p1 = players_one_lose.pop(0)
            p2 = players_one_lose.pop(0)
            match_schedule.append((p1, p2))
            total_matches_num += 1 
        if len(players_one_lose) == 1:
            match_schedule.append((players_one_lose.pop(), 'N/A'))
        return match_schedule, total_matches_num

def double_elimination_random(win_matrix, player_information, round_num, matches_num, finish_all_rounds):
    """
    双淘汰赛：选手需输两场比赛才被淘汰。
    
    参数：
      - win_matrix, player_information 同上，其中 player_information 需含 'Lose_Time' 列（初始均为 0）。
      - round_num: 控制比赛轮数；若采用场次数控制，则设置为 None。
      - matches_num: 控制比赛总场数；若采用轮次数控制，则设置为 None。
      - finish_all_rounds: 若为 True，则不考虑 rounds_num 与 matches_num。
    
    返回：
      - de_information: 双淘汰赛结束后的选手信息 DataFrame。
      - ranked_de_information: 排名后的结果 DataFrame。
    """
    de_information = player_information.copy()
    de_information["Lose_Time"] = 0
    total_matches_num = 0

    if finish_all_rounds:
        matches_num = 1000
        round_num = None

    if round_num is None:
        round_num = 1000

    for i in range(round_num):
        if len(de_information[de_information['Lose_Time'] == 2]["Player"].tolist()) == len(de_information)-1:
            # print("比赛已达最大轮次，结束。")
            break
        if i == 0:
            if matches_num is not None and total_matches_num == matches_num:
                break
            one_round_list, total_matches_num = generate_double_elimination_pairs(de_information, True, total_matches_num, matches_num)
            for p1, p2 in one_round_list:
                de_information = win_judge_with_lose_time(win_matrix, p1, p2, de_information)
        else:
            if matches_num is not None and total_matches_num == matches_num:
                break
            one_round_list, total_matches_num = generate_double_elimination_pairs(de_information, False, total_matches_num, matches_num)
            for p1, p2 in one_round_list:
                de_information = win_judge_with_lose_time(win_matrix, p1, p2, de_information)
    ranked_de_information = rank_players(de_information)
    return de_information, ranked_de_information

"""
加权循环赛
选手之间的每场比赛可能具有不同的权重。例如，前几名的选手之间的比赛可能比后几名选手之间的比赛更重要。权重通常会影响选手的得分或者比赛结果。
"""
def generate_weighted_match_pairs(num_rows):
    is_odd = num_rows % 2 != 0
    if is_odd:
        num_rows += 1  # 增加虚拟玩家

    # 初始化选手列表：若原始为奇数，则最后一个为 'N/A'
    players = list(range(1, num_rows+1)) if not is_odd else list(range(1, num_rows)) + ['N/A']
    # random.shuffle(players)
    match_schedule = []
    # 为每场比赛分配一个权重
    # 生成一个字典，键为选手编号，值为该选手的比赛权重
    weights = np.linspace(np.sqrt(num_rows), 1, num_rows-1)  # 比赛权重从1到人数开根线性变化
    weights_dict = {players[i]: weights[i] for i in range(num_rows-1)}
    for _ in range(num_rows - 1):
        pairs = []
        for i in range(len(players) // 2):
            if players[len(players) - 1 - i] == 'N/A':
                pairs.append((players[i], players[len(players) - 1 - i], weights_dict[players[i]]))
            elif players[i] == 'N/A':
                pairs.append((players[i], players[len(players) - 1 - i], weights_dict[players[len(players) - 1 - i]]))
            else:
                pairs.append((players[i], players[len(players) - 1 - i], (weights_dict[players[i]]+weights_dict[players[len(players) - 1 - i]])/2))
        match_schedule.append(pairs)
        # 轮转：保持第一个选手不变，其余选手轮转
        players = [players[0]] + players[1:][1:] + [players[1]]
    return match_schedule


def weighted_round_robin(win_matrix, player_information, rounds_num, matches_num, finish_all_rounds):
    weighted_information = player_information.copy()
    num_players = weighted_information.shape[0]
    
    match_schedule = generate_weighted_match_pairs(num_players)

    if finish_all_rounds:
        # 保留所有生成的轮次
        pass
    elif matches_num is None:
        if rounds_num > num_players:
            # print("轮次数超过最多可能轮次数，自动调整。")
            rounds_num = num_players
        match_schedule = match_schedule[:rounds_num]
    elif rounds_num is None:
        # if matches_num < math.ceil(num_rows / 2):
        #     print("比赛场次数小于选手数量，无法完成一轮。")
        truncated_schedule = []
        for round_pairs in match_schedule:
            round_truncated = []
            for pair in round_pairs:
                if matches_count < matches_num:
                    round_truncated.append(pair)
                    matches_count += 1
            if round_truncated:
                truncated_schedule.append(round_truncated)
        match_schedule = truncated_schedule

    for round_pairs in match_schedule:
        for p1, p2, weight in round_pairs:
            if p1 == 'N/A':
                weighted_information.loc[weighted_information['Player'] == p2, 'Score'] += weight
                weighted_information.loc[weighted_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
            elif p2 == 'N/A':
                weighted_information.loc[weighted_information['Player'] == p1, 'Score'] += weight
                weighted_information.loc[weighted_information['Player'] == p1, 'Defeated_Opponents'].values[0].append(p2)
            else:
                win_rate = win_matrix[p1-1, p2-1]
                if np.random.rand() < win_rate:
                    weighted_information.loc[weighted_information['Player'] == p1, 'Score'] += weight
                    weighted_information.loc[weighted_information['Player'] == p1, 'Defeated_Opponents'].values[0].append(p2)
                else:
                    weighted_information.loc[weighted_information['Player'] == p2, 'Score'] += weight
                    weighted_information.loc[weighted_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
            
    ranked_information = rank_players(weighted_information)
    
    return player_information, ranked_information

########################################################################
# 分组赛 + 淘汰赛相关函数
########################################################################

def generate_groups(num_players, num_groups):
    """
    将总选手随机分为 num_groups 组。
    
    参数：
      - num_players: 整数，总选手数量。
      - num_groups: 整数，分组数。
    
    返回：
      - groups: 列表，每个元素为一个分组（选手编号列表）。
    """
    players = list(range(1, num_players + 1))
    random.shuffle(players)
    groups = [players[i::num_groups] for i in range(num_groups)]
    return groups

def knockout_stage(group_results, win_matrix, rr_player_information, knockout_stage_rounds_num, knockout_stage_matches_num):
    """
    淘汰赛阶段：根据各小组循环赛结果进入单败淘汰赛。
    
    说明：
      - 从每个小组中选取前两名进入淘汰赛。
      - 使用 generate_random_match_pairs 生成对阵表（该函数已调整为不修改原列表）。
      - 每场比赛由 win_judge_return_winner 判断胜者，输家从淘汰选手列表中移除。
    
    参数：
      - group_results: 列表，每个元素为一组小组赛的结果 DataFrame。
      - win_matrix: 胜负判断矩阵。
      - rr_player_information: 循环赛阶段合并后的选手信息 DataFrame。
      - knockout_stage_rounds_num: 淘汰赛允许的轮次数；若采用场次数控制，则为 None。
      - knockout_stage_matches_num: 淘汰赛允许的总比赛场次数；若采用轮次数控制，则为 None。
    
    返回：
      - rr_player_information: 淘汰赛结束后更新的选手信息 DataFrame。
    """
    knockout_players = []
    rounds_count = 0
    match_count = 0

    # 取各组前两名
    for group in group_results:
        # 如果每组人数不足两名
        if len(group) < 2:
            knockout_players.append(group.iloc[0])
        else:
            knockout_players.append(group.iloc[0])
            knockout_players.append(group.iloc[1])

    knockout_information = pd.DataFrame(knockout_players)
    players = knockout_information['Player'].tolist()
    players_copy = players.copy()

    while len(players_copy) > 1:
        match_schedule = generate_random_match_pairs(players_copy)
        for p1, p2 in match_schedule:
            loser = win_judge_return_winner(win_matrix, p1, p2, rr_player_information)
            if loser in players_copy:
                players_copy.remove(loser)
        rounds_count += 1
        if knockout_stage_rounds_num is not None and rounds_count == knockout_stage_rounds_num:
            break
        match_count += len(match_schedule)
        if knockout_stage_matches_num is not None and match_count == knockout_stage_matches_num:
            break
    return rr_player_information

def map_defeated_opponents(defeated_list, new_to_old):
    """
    将 defeated_list 中的每个元素根据 new_to_old 映射回原始编号。
    
    参数：
      - defeated_list: 列表，存储已击败对手的编号。
      - new_to_old: 字典，映射新编号到原始编号。
    
    返回：
      - 映射后的列表。
    """
    if not defeated_list:
        return []
    return [new_to_old.get(x, x) for x in defeated_list]

def total_rr_matches(win_matrix, group_num):
    """
    计算将总人数分为 group_num 组后，各组内部循环赛的总比赛场数之和。
    
    说明：
      - 对于 n 人组，其内部比赛场数为 n*(n-1)/2。
      - 当不能均分时，部分组人数为 base+1，其余为 base。
    
    参数：
      - win_matrix: 用于获取总人数（行数）。
      - group_num: 分组数。
    
    返回：
      - total_matches: 整数，总比赛场数之和。
    """
    total_players = win_matrix.shape[0]
    base = total_players // group_num
    remainder = total_players % group_num
    total_matches = 0
    for _ in range(remainder):
        group_size = base + 1
        total_matches += group_size * (group_size - 1) // 2
    for _ in range(group_num - remainder):
        group_size = base
        total_matches += group_size * (group_size - 1) // 2
    return total_matches

def rr_knockout(win_matrix, player_information, rounds_num, matches_num, finish_all_rounds):
    """
    分组赛 + 淘汰赛：
      - 循环赛阶段：各分组内部进行循环赛，各组比赛轮次相同，每轮结果保留组号。
      - 淘汰赛阶段：根据各组排名进入单败淘汰赛。
    这里的轮次数量分为两部分：循环赛的轮次数量和淘汰赛的比赛场次数量。其中，循环赛的轮次数量为一个小组的轮次数量，因为每个小组的人数是相同的，我们认为一轮比赛包括所有的小组的该轮。
    
    参数：
      - win_matrix, player_information 同前。
      - rounds_num: 循环赛阶段比赛轮数；若采用场次数控制，则为 None。
      - matches_num: 循环赛阶段比赛场次数；若采用轮次数控制，则为 None。
      - finish_all_rounds: 若为 True，则忽略 rounds_num 与 matches_num。
    
    返回：
      - 原始 player_information 与综合（循环赛 + 淘汰赛）后的最终排名结果 DataFrame。
    """
    group_results = []
    rr_player_information_list = []
    is_knockout = True
    rrknockout_information = player_information.copy()
    group_num = 4
    groups = generate_groups(len(rrknockout_information), group_num)

    max_rounds_num_in_rr = math.ceil(win_matrix.shape[0] / group_num) - 1
    max_matches_num_in_rr = total_rr_matches(win_matrix, group_num)
    if rounds_num is not None and rounds_num < max_rounds_num_in_rr:
        # print("轮次数小于循环赛阶段最多可能轮次数。")
        is_knockout = False
    if matches_num is not None and matches_num < max_matches_num_in_rr:
        # print("场次数小于循环赛阶段最多可能场次数。")
        is_knockout = False
    if finish_all_rounds:
        rounds_num = None
        matches_num = None
        knockout_stage_rounds_num = None
        knockout_stage_matches_num = None

    for group_id, group in enumerate(groups, start=1):
        group_info = rrknockout_information[rrknockout_information['Player'].isin(group)].copy()
        group_info = group_info.sort_values(by='Player', ascending=True).reset_index(drop=True)
        group = sorted(group)
        group_indices = sorted([x - 1 for x in group])
        sub_win_matrix = win_matrix[np.ix_(group_indices, group_indices)]
        new_to_old = {new_id: orig for new_id, orig in enumerate(group, start=1)}
        old_to_new = {v: k for k, v in new_to_old.items()}
        group_info['Player'] = group_info['Player'].map(old_to_new)
        _, ranked_rr_group_information = robin_round(sub_win_matrix, group_info, rounds_num, matches_num, finish_all_rounds)
        ranked_rr_group_information['Player'] = ranked_rr_group_information['Player'].map(new_to_old)
        if 'Defeated_Opponents' in ranked_rr_group_information.columns:
            ranked_rr_group_information['Defeated_Opponents'] = ranked_rr_group_information['Defeated_Opponents'].apply(
                lambda lst: map_defeated_opponents(lst, new_to_old)
            )
        ranked_rr_group_information['Group'] = group_id
        rr_player_information_list.append(ranked_rr_group_information)
        group_results.append(ranked_rr_group_information)
    
    rr_player_information = pd.concat(rr_player_information_list, ignore_index=True)
    if is_knockout:
        if rounds_num is not None:
            knockout_stage_rounds_num = rounds_num - max_rounds_num_in_rr
            knockout_stage_matches_num = None
        elif matches_num is not None:
            knockout_stage_matches_num = matches_num - max_matches_num_in_rr
            knockout_stage_rounds_num = None
        rr_player_information = knockout_stage(group_results, win_matrix, rr_player_information, knockout_stage_rounds_num, knockout_stage_matches_num)
    ranked_information = rank_players(rr_player_information)
    return player_information, ranked_information

########################################################################
# 阶梯赛（Ladder Tournament）相关函数
# 阶梯赛制的round思路：每一个round完成一次从末尾选手到第一位选手的循环
########################################################################

def ladder_tournament(win_matrix, player_information, round_num, matches_num, finish_all_rounds):
    """
    阶梯赛：选手依次挑战比自己排名更高者，挑战成功则交换位置。
    
    参数：
      - win_matrix, player_information 同前。
      - round_num: 指定比赛轮数；若采用场次数控制，则为 None。
      - matches_num: 指定比赛场次数；若采用轮次数控制，则为 None。
      - finish_all_rounds: 若为 True，则只执行一轮比赛，忽略其它控制参数。
    
    返回：
      - 原始 player_information 与经过阶梯赛后的排名结果 DataFrame。
    """
    ladder_information = player_information.copy()
    num_players = len(ladder_information)
    played_matches = 0
    actual_rounds = 0

    if finish_all_rounds:
        for i in range(num_players - 1, 0, -1):
            played_matches += 1
            challenger = ladder_information.iloc[i]
            target = ladder_information.iloc[i-1]
            winner = win_judge_return_winner(win_matrix, challenger['Player'], target['Player'], ladder_information)
            if winner == challenger['Player']:
                ladder_information.iloc[i], ladder_information.iloc[i-1] = ladder_information.iloc[i-1], ladder_information.iloc[i]
        ranked_information = rank_players(ladder_information)
        return player_information, ranked_information

    if round_num is not None:
        for _ in range(round_num):
            for i in range(num_players - 1, 0, -1):
                played_matches += 1
                challenger = ladder_information.iloc[i]
                target = ladder_information.iloc[i-1]
                winner = win_judge_return_winner(win_matrix, challenger['Player'], target['Player'], ladder_information)
                if winner == challenger['Player']:
                    ladder_information.iloc[i], ladder_information.iloc[i-1] = ladder_information.iloc[i-1], ladder_information.iloc[i]
            actual_rounds += 1
    elif matches_num is not None:
        while played_matches < matches_num:
            round_occurred = False
            for i in range(num_players - 1, 0, -1):
                if played_matches >= matches_num:
                    break
                round_occurred = True
                played_matches += 1
                challenger = ladder_information.iloc[i]
                target = ladder_information.iloc[i-1]
                winner = win_judge_return_winner(win_matrix, challenger['Player'], target['Player'], ladder_information)
                if winner == challenger['Player']:
                    ladder_information.iloc[i], ladder_information.iloc[i-1] = ladder_information.iloc[i-1], ladder_information.iloc[i]
            if not round_occurred:
                break
            actual_rounds += 1

    ranked_information = rank_players(ladder_information)
    return player_information, ranked_information