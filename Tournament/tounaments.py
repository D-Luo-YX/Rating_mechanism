import numpy as np
import pandas as pd
import random
import numpy as np
from .win_judge import win_judge_return_winner, generate_random_match_pairs, one_round_match, rank_players
###########################################################
######################## 赛制 ##############################
###########################################################
"""
循环赛比赛
"""
def generate_match_pairs(num_rows):
    """
    生成基于轮转思想的对阵表，支持偶数和奇数数量的玩家。每个选手都会和其他选手对阵一次，适用于循环赛比赛。
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

def robin_round(win_matrix, player_information):
    rr_information = player_information.copy()
    num_rows = rr_information.shape[0]

    match_schedule = generate_match_pairs(num_rows)
    for i in range(num_rows-1):
        rr_information = one_round_match(win_matrix, rr_information, match_schedule[i])

    ranked_rr_information = rank_players(rr_information)

    return player_information, ranked_rr_information


"""
瑞士轮比赛
"""
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
    return player_information, ranked_sr_information

"""
双淘汰赛比赛
"""
def double_elimination_random(win_matrix, player_information):
    de_information = player_information.copy()

    # 初始化选手列表
    winners_bracket = player_information["Player"].tolist()
    round_results = []  # 记录每轮的胜者和负者
    round_num = 0

    # 胜者组比赛：直到剩下1名选手
    while len(winners_bracket) > 1:
        winners_match_schedule = generate_random_match_pairs(winners_bracket)
        winners_this_round = []
        losers_this_round = []
        round_result = {'type':'winner_group', 'round': round_num + 1, 'winners': [], 'losers': []}

        for p1, p2 in winners_match_schedule:
            winner = win_judge_return_winner(win_matrix, p1, p2, de_information)
            winners_this_round.append(winner)

            if winner == p1:
                losers_this_round.append(p2)
                round_result['winners'].append(p1)
                round_result['losers'].append(p2)
            else:
                losers_this_round.append(p1)
                round_result['winners'].append(p2)
                round_result['losers'].append(p1)

        winners_bracket = winners_this_round

        # 将当前轮次的败者加入所有轮次结果
        round_results.append(round_result)  # 存储本轮结果
        round_num += 1

    # 败者组比赛：按轮次进行
    competitors_this_round = [] # 记录当前轮次的选手 
    winners_by_round = []  # 记录每轮的败者
    losers_by_round = []  # 记录每轮的胜者

    rounds_in_winner_group = round_num
    round_num = 0

    # 根据每轮的败者进行配对
    while winners_by_round ==[] or len(winners_by_round[-1]) != 1:
        # 第一轮的败者组比赛随机配对
        if round_results[-1]['type'] == 'winner_group':
            competitors_this_round = round_results[round_num]['losers']  # 获取第一轮比赛的败者
            match_schedule = generate_random_match_pairs(competitors_this_round)
        # 如果胜者组的所有losers已经为空了，则直接将负者组胜利的选手继续比赛
        elif round_results[rounds_in_winner_group - 1]['losers'] == []:
            competitors_this_round = round_results[-1]['winners_of_loser_group']
            save_competitors_this_round = competitors_this_round.copy()
            match_schedule = generate_random_match_pairs(competitors_this_round)
            round_results[-1]['winners_of_loser_group'] = save_competitors_this_round
        # 如果胜者组的loser还有选手，则和胜者组下一轮次的败者配对
        else:
            competitors_this_round = round_results[-1]['winners_of_loser_group']+round_results[round_num]['losers']
            match_schedule = generate_random_match_pairs(competitors_this_round)
            round_results[round_num]['losers'] = []
        # elif len(round_results[-1]['losers_of_loser_group']) > len(round_results[round_num]['losers']):
        #     match_schedule = round_results[-1]['winners_of_loser_group']
        #     match_schedule = generate_random_match_pairs(match_schedule)

        winners_this_round = []
        round_result = {'type':'loser_group', 'round': round_num + 1, 'winners_of_loser_group': [], 'losers_of_loser_group': []}

        for p1, p2 in match_schedule:
            winner = win_judge_return_winner(win_matrix, p1, p2, de_information)
            winners_this_round.append(winner)

            if winner == p1:
                round_result['winners_of_loser_group'].append(p1)
                round_result['losers_of_loser_group'].append(p2)
            else:
                round_result['winners_of_loser_group'].append(p2)
                round_result['losers_of_loser_group'].append(p1)

        # 更新败者组的败者和败者组的胜利者并存储
        losers_by_round.append(round_result['losers_of_loser_group'])
        winners_by_round.append(round_result['winners_of_loser_group'])
        round_results.append(round_result)  # 存储本轮结果

        round_num += 1

    # 最终决赛
    final_winner_of_winner_group = round_results[rounds_in_winner_group-1]['winners']
    final_winner_of_loser_group = round_results[-1]['winners_of_loser_group']
    final_match = generate_random_match_pairs(final_winner_of_winner_group + final_winner_of_loser_group)
    final_result = None
    for p1, p2 in final_match:
        final_result = win_judge_return_winner(win_matrix, p1, p2, de_information)

    ranked_de_information = rank_players(de_information)

    return de_information, ranked_de_information

"""
加权循环赛
选手之间的每场比赛可能具有不同的权重。例如，前几名的选手之间的比赛可能比后几名选手之间的比赛更重要。权重通常会影响选手的得分或者比赛结果。
"""
def generate_weighted_match_pairs(num_players):
    players = list(range(1, num_players + 1))
    match_schedule = []
    
    # 为每场比赛分配一个权重
    weights = np.linspace(10, 1, num_players-1)  # 比赛权重从1到10线性变化
    
    for i in range(num_players):
        for j in range(i + 1, num_players):
            match_schedule.append((players[i], players[j], weights[i]))
    
    return match_schedule


def weighted_round_robin(win_matrix, player_information):
    weighted_information = player_information.copy()
    num_players = weighted_information.shape[0]
    
    match_schedule = generate_weighted_match_pairs(num_players)

    for p1, p2, weight in match_schedule:
        win_rate = win_matrix[p1-1, p2-1]
        if np.random.rand() < win_rate:
            weighted_information.loc[weighted_information['Player'] == p1, 'Score'] += weight
            weighted_information.loc[weighted_information['Player'] == p1, 'Defeated_Opponents'].values[0].append(p2)
        else:
            weighted_information.loc[weighted_information['Player'] == p2, 'Score'] += weight
            weighted_information.loc[weighted_information['Player'] == p2, 'Defeated_Opponents'].values[0].append(p1)
    
    ranked_information = rank_players(weighted_information)
    
    return player_information, ranked_information

"""
分组赛+淘汰赛
将选手分为若干组进行小组赛，按小组排名前几名进入淘汰赛。类似世界杯或欧冠。
"""
# 生成分组
def generate_groups(num_players, num_groups):
    players = list(range(1, num_players + 1))
    random.shuffle(players)
    groups = [players[i::num_groups] for i in range(num_groups)]
    return groups

# 淘汰赛阶段
def knockout_stage(group_results, win_matrix, rr_player_information):
    """
    淘汰赛阶段，根据小组赛结果进行单败淘汰赛。
    """
    knockout_players = []

    # 从每个小组中选出前2名进入淘汰赛
    for group in group_results:
        knockout_players.append(group.iloc[0])  # 前1名
        knockout_players.append(group.iloc[1])  # 前2名

    knockout_information = pd.DataFrame(knockout_players)
    
    # 开始单败淘汰赛
    # initial_sorted_knockout = knockout_information.sort_values(by='Score', ascending=False).reset_index(drop=True)  # 初始排序
    players = knockout_information['Player'].tolist()
    players_copy = players.copy()
    while len(players_copy) > 1:
        # 第一轮比赛
        if players != []:
            match_schedule = generate_random_match_pairs(players)
        else:
            players_copy_copy = players_copy.copy()
            match_schedule = generate_random_match_pairs(players_copy_copy)
    
        for p1, p2 in match_schedule:
            loser = win_judge_return_winner(win_matrix, p1, p2, rr_player_information)
            players_copy.remove(loser)

    return rr_player_information

def map_defeated_opponents(defeated_list, new_to_old):
    """
    将 defeated_list 中的每个元素根据 new_to_old 映射回原始编号
    """
    # 如果 defeated_list 为空，直接返回空列表
    if not defeated_list:
        return []
    return [new_to_old.get(x, x) for x in defeated_list]

def rr_knockout(win_matrix, player_information):
    group_results = []
    rr_player_information_list = []  # 用于保存每个小组的循环赛结果

    rrgs_information = player_information.copy()
    groups = generate_groups(len(rrgs_information), 4)
    for group_id, group in enumerate(groups, start=1):
        # 提取本组选手数据
        group_info = rrgs_information[rrgs_information['Player'].isin(group)].copy()
        group_info = group_info.sort_values(by='Player', ascending=True).reset_index(drop=True)
        group = sorted(group)

        # 该小组对应的子胜率矩阵
        group_indices = [x - 1 for x in group]
        group_indices.sort()
        sub_win_matrix = win_matrix[np.ix_(group_indices, group_indices)]

        # 创建映射，新的连续编号 <--> 原始编号
        new_to_old = {new_id: orig for new_id, orig in enumerate(group, start=1)}
        old_to_new = {v: k for k, v in new_to_old.items()}
        
        # 将本组DataFrame的Player列转换为新编号
        group_info['Player'] = group_info['Player'].map(old_to_new)
        
        # 调用robin_round进行循环赛
        _, ranked_rr_group_information = robin_round(sub_win_matrix, group_info)
        
        # 将重新编号的Player列映射回原始编号
        ranked_rr_group_information['Player'] = ranked_rr_group_information['Player'].map(new_to_old)
        if 'Defeated_Opponents' in ranked_rr_group_information.columns:
            ranked_rr_group_information['Defeated_Opponents'] = ranked_rr_group_information['Defeated_Opponents'].apply(
                lambda lst: map_defeated_opponents(lst, new_to_old)
            )
        
        # 保存本组的结果到列表中
        rr_player_information_list.append(ranked_rr_group_information)
        group_results.append(ranked_rr_group_information)
    
    # 合并所有小组的循环赛结果
    rr_player_information = pd.concat(rr_player_information_list, ignore_index=True)
    
    # 进入淘汰赛阶段（假设 knockout_stage 接收合并后的 DataFrame）
    rr_player_information_knockout_stage = knockout_stage(group_results, win_matrix, rr_player_information)
    ranked_information = rank_players(rr_player_information_knockout_stage)

    return player_information, ranked_information

"""
阶梯赛
阶梯赛是一种选手逐步上升的赛制，每个选手必须挑战比自己排名更高的选手，成功后可以晋升到更高的排名。
"""
def ladder_tournament(win_matrix, player_information):
    ladder_information = player_information.copy()

    num_players = len(ladder_information)
    # 从后往前
    for i in range(num_players - 1, 0, -1):
        # 每一轮选择两个选手进行比赛：当前选手与挑战者
        challenger = ladder_information.iloc[i]
        target = ladder_information.iloc[i-1]

        # 获取比赛结果，判断哪一方胜出
        winner = win_judge_return_winner(win_matrix, challenger['Player'], target['Player'], ladder_information)

        if winner == challenger['Player']:
            # 挑战成功，换位
            ladder_information.iloc[i], ladder_information.iloc[i-1] = ladder_information.iloc[i-1], ladder_information.iloc[i]

    # 返回最终排序
    ranked_information = rank_players(ladder_information)
    
    return player_information, ranked_information
