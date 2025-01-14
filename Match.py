import numpy as np
import pandas as pd



def strength_list(num_players=32, strengths_type='Uniform', simulate_and_1_flag = False):
    strengths = []
    if simulate_and_1_flag: simulate_num = int(num_players*2)
    else: simulate_num = int(num_players)
    if strengths_type == 'Uniform':
        # 均匀分布
        strengths = np.random.uniform(0, 1, simulate_num)

    #幂律分布
    elif strengths_type == 'PL':
        a = 0.5
        strengths = np.random.power(a, simulate_num)

    # 正态分布 重映射正态分布
    elif strengths_type == 'Normal':
        normal_data = np.random.randn(simulate_num)
        strengths = 1 / (1 + np.exp(-normal_data))

    strengths_df = pd.DataFrame(strengths, columns=["Strength"])
    strengths_df['Player'] = strengths_df.index

    sorted_strengths_df = strengths_df.sort_values(by="Strength", ascending=False).reset_index(drop=True)

    sorted_strengths = sorted_strengths_df['Strength'].values

    # 创建 DataFrame，记录选手信息
    strengths_df = pd.DataFrame({
        "Player": range(1, simulate_num + 1),  # 选手编号从 1 开始
        "Strength": sorted_strengths,
        "Score": 0,  # 初始化分数为 0
        "Defeated_Opponents": [[] for _ in range(simulate_num)]  # 初始化为一个空列表
    })

    # 按实力排序
    sorted_strengths_df = strengths_df.sort_values(by="Strength", ascending=False).reset_index(drop=True)

    return sorted_strengths_df


def win_judge(theta, p1, p2, player_information):
    """
    通过 theta 和两个选手编号 p1、p2 判断 p1 是否战胜 p2
    参数：
    - theta: 控制胜率敏感度的参数
    - p1, p2: 两个选手的编号
    - player_information: 包含选手信息的 DataFrame，需包含 'Player' 和 'Strength' 列

    返回：
    - updated_player_information: 更新后的选手信息 DataFrame
    """
    # 获取选手 p1 和 p2 的实力
    s1 = player_information.loc[player_information['Player'] == p1, 'Strength'].values[0]
    s2 = player_information.loc[player_information['Player'] == p2, 'Strength'].values[0]

    # 计算 p1 战胜 p2 的胜率
    win_rate = (s1 ** theta) / (s1 ** theta + s2 ** theta)

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

def one_round_match(player_information, schedule, theta):
    for p1, p2 in schedule:
        player_information = win_judge(theta, p1, p2, player_information)
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

def robin_round(player_information,theta):
    rr_information = player_information.copy()
    num_rows = rr_information.shape[0]

    match_schedule = generate_match_pairs(num_rows)
    for i in range(num_rows-1):
        one_round_match(rr_information, match_schedule[i],theta)

    ranked_rr_information = rank_players(rr_information)

    return rr_information, ranked_rr_information

def swiss_round(player_information,theta):

    sw_information = player_information.copy()



    ranked_sw_information = rank_players(sw_information)
    return sw_information, ranked_sw_information

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
    - player_information: 初始选手信息的 DataFrame，按实力排序
    - ranked_information: 比赛结果排名后的 DataFrame，按得分排名

    返回：
    - ndcg_spearman_score: 引入 NDCG 思想的 Spearman 相关系数
    """
    # 提取选手编号列表
    initial_players = player_information['Player'].tolist()
    ranked_players = ranked_information['Player'].tolist()

    # 将初始排名和比赛排名转换为 Pandas Series，索引为 Player
    initial_rank = pd.Series(range(1, len(initial_players) + 1), index=initial_players)
    result_rank = pd.Series(range(1, len(ranked_players) + 1), index=ranked_players)

    # 计算排名差 |d_i| 和折扣因子 1 / log2(i + 1)
    n = len(player_information)
    d_squared_sum = 0
    for i, player in enumerate(initial_rank.index):
        d_i = abs(initial_rank[player] - result_rank[player])  # 计算选手实际排名的差异
        discount = 1 / np.log2(i + 2)  # 折扣因子，i + 2 因为 i 从 0 开始
        d_squared_sum += (d_i ** 2) * discount

    # 计算理想情况下的归一化因子
    max_d_squared_sum = np.sum([(i + 1) ** 2 / np.log2(i + 2) for i in range(n)])

    # 计算引入 NDCG 思想的 Spearman 相关系数
    ndcg_spearman_score = 1 - (6 * d_squared_sum) / max_d_squared_sum


    return ndcg_spearman_score

if __name__ == "__main__":
    Initial_Player_DataFrame = strength_list(num_players=32, strengths_type='Uniform', simulate_and_1_flag=False)
    rr, ranked_rr = robin_round(Initial_Player_DataFrame, 0.88)

    spearman_score_rr = calculate_coefficient(rr,ranked_rr)
    ndcg_spearman_score_rr = calculate_ndcg_coefficient(rr,ranked_rr)

    print(f"Spearman correlation coefficient: {spearman_score_rr}")
    print(f"NDCG-Spearman correlation coefficient: {ndcg_spearman_score_rr}")
    # processing_information = win_judge(theta=0.88,p1=1,p2=10,player_information=Initial_Player_DataFrame)
    # print(processing_information)
