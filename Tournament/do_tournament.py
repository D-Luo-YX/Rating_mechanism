import os
import numpy as np
import pandas as pd
import random
from matplotlib import rcParams
from .coefficient import calculate_Spearman_coefficient, calculate_ndcg_Spearman_coefficient
from .normalized_tounaments import robin_round, swiss_round, double_elimination_random, weighted_round_robin, rr_knockout, ladder_tournament
# from coefficient import calculate_Spearman_coefficient, calculate_ndcg_Spearman_coefficient

# from normalized_tounaments import robin_round, swiss_round, double_elimination_random, weighted_round_robin, rr_knockout, ladder_tournament


def standard_matrix(rows, cols):
    """
    用真实数据的胜负率生成矩阵，所有值固定为 0.5。
    """
    return np.full((rows, cols), 0.5)

class PlayerData:
    def __init__(self, player_count):
        self.df = pd.DataFrame({
            "Player": range(1, player_count + 1),
            "Score": [0.0] * player_count,
            "Defeated_Opponents": [[] for _ in range(player_count)]
        })
    def get_copy(self):
        return self.df.copy(deep=True)

def tournament_correlation(match_name, distribution_type, win_matrix, iterations, datatype, param_type, param_values, save_path):
    """
    模拟多个赛制下的比赛，并返回各参数值下各赛制平均 Spearman 系数的结果，同时将每个参数计算结果以单行形式追加保存到 CSV 文件中，
    以防止因运行中断而丢失中间结果。
    
    参数：
      - match_name: 比赛名称
      - distribution_type: 选手实力分布类型
      - win_matrix: 胜负矩阵
      - iterations: 每个参数值下模拟的迭代次数
      - datatype: 数据来源类型说明（"Real Data" 或 "Simulation"）
      - param_type: "rounds" 表示以轮次数控制；"matches" 表示以场次数控制；"finish_all_rounds" 表示直到所有轮次结束
      - param_values: 要测试的 rounds_num 或 matches_num 参数值列表
      - save_path: 保存 CSV 文件的目录路径

    返回：
      - results_df: 包含所有模拟结果的 DataFrame（读取保存的 CSV 文件）
    """

    # 如果是完成所有的轮次，则不需要参数值
    if param_type == "finish_all_rounds":
        rounds_num = None
        matches_num_val = None
        finish_all_rounds_flag = True

        # spearman 系数求和
        sum_spearman_rr = 0.0
        sum_spearman_sr = 0.0
        sum_spearman_de = 0.0
        sum_spearman_weighted = 0.0
        sum_spearman_rr_knockout = 0.0
        sum_spearman_ladder = 0.0
        # ndcg_spearman 系数求和
        sum_ndcg_spearman_rr = 0.0
        sum_ndcg_spearman_sr = 0.0
        sum_ndcg_spearman_de = 0.0
        sum_ndcg_spearman_weighted = 0.0
        sum_ndcg_spearman_rr_knockout = 0.0
        sum_ndcg_spearman_ladder = 0.0
        
        for i in range(iterations):
            player_data = PlayerData(win_matrix.shape[0])
            rr_initial, rr_ranked = robin_round(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
            player_data = PlayerData(win_matrix.shape[0])
            sr_initial, sr_ranked = swiss_round(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
            player_data = PlayerData(win_matrix.shape[0])
            de_initial, de_ranked = double_elimination_random(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
            player_data = PlayerData(win_matrix.shape[0])
            weighted_initial, weighted_ranked = weighted_round_robin(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
            player_data = PlayerData(win_matrix.shape[0])
            rr_knockout_initial, rr_knockout_ranked = rr_knockout(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
            player_data = PlayerData(win_matrix.shape[0])
            ladder_initial, ladder_ranked = ladder_tournament(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
            
            sum_spearman_rr += calculate_Spearman_coefficient(rr_initial, rr_ranked)
            sum_spearman_sr += calculate_Spearman_coefficient(sr_initial, sr_ranked)
            sum_spearman_de += calculate_Spearman_coefficient(de_initial, de_ranked)
            sum_spearman_weighted += calculate_Spearman_coefficient(weighted_initial, weighted_ranked)
            sum_spearman_rr_knockout += calculate_Spearman_coefficient(rr_knockout_initial, rr_knockout_ranked)
            sum_spearman_ladder += calculate_Spearman_coefficient(ladder_initial, ladder_ranked)

            sum_ndcg_spearman_rr += calculate_ndcg_Spearman_coefficient(rr_initial, rr_ranked)
            sum_ndcg_spearman_sr += calculate_ndcg_Spearman_coefficient(sr_initial, sr_ranked)
            sum_ndcg_spearman_de += calculate_ndcg_Spearman_coefficient(de_initial, de_ranked)
            sum_ndcg_spearman_weighted += calculate_ndcg_Spearman_coefficient(weighted_initial, weighted_ranked)
            sum_ndcg_spearman_rr_knockout += calculate_ndcg_Spearman_coefficient(rr_knockout_initial, rr_knockout_ranked)
            sum_ndcg_spearman_ladder += calculate_ndcg_Spearman_coefficient(ladder_initial, ladder_ranked)
        
            print(f"Match: {match_name}; Distribution: {distribution_type}; DataType: {datatype}; Param:finish_all; Iteration: {i+1}/{iterations} finished.")

        # 计算平均值
        avg_spearman_rr = sum_spearman_rr / iterations
        avg_spearman_sr = sum_spearman_sr / iterations
        avg_spearman_de = sum_spearman_de / iterations
        avg_spearman_weighted = sum_spearman_weighted / iterations
        avg_spearman_rr_knockout = sum_spearman_rr_knockout / iterations
        avg_spearman_ladder = sum_spearman_ladder / iterations

        avg_ndcg_spearman_rr = sum_ndcg_spearman_rr / iterations
        avg_ndcg_spearman_sr = sum_ndcg_spearman_sr / iterations
        avg_ndcg_spearman_de = sum_ndcg_spearman_de / iterations
        avg_ndcg_spearman_weighted = sum_ndcg_spearman_weighted / iterations
        avg_ndcg_spearman_rr_knockout = sum_ndcg_spearman_rr_knockout / iterations
        avg_ndcg_spearman_ladder = sum_ndcg_spearman_ladder / iterations
        
        row = {
            "Match": match_name,
            "Distribution Type": distribution_type,
            "Data Type": datatype,
            "Parameter Type": param_type,
            "Parameter": 'None',
            "Finish all rounds": 'True',
            "Spearman Round Robin": avg_spearman_rr,
            "Spearman Swiss Round": avg_spearman_sr,
            "Spearman Double Elimination": avg_spearman_de,
            "Spearman Weighted Round Robin": avg_spearman_weighted,
            "Spearman Round Robin Knockout": avg_spearman_rr_knockout,
            "Spearman Ladder Tournament": avg_spearman_ladder,
            "NDCG Spearman Round Robin": avg_ndcg_spearman_rr,
            "NDCG Spearman Swiss Round": avg_ndcg_spearman_sr,
            "NDCG Spearman Double Elimination": avg_ndcg_spearman_de,
            "NDCG Spearman Weighted Round Robin": avg_ndcg_spearman_weighted,
            "NDCG Spearman Round Robin Knockout": avg_ndcg_spearman_rr_knockout,
            "NDCG Spearman Ladder Tournament": avg_ndcg_spearman_ladder
        }
        
        # 将该行结果追加到 CSV 文件
        temp_df = pd.DataFrame([row])
        # 如果文件不存在，则写入表头；否则追加写入且不写入表头
        temp_df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))


    # 否则，需要遍历参数值列表
    else:
        for param in param_values:
            sum_spearman_rr = 0.0
            sum_spearman_sr = 0.0
            sum_spearman_de = 0.0
            sum_spearman_weighted = 0.0
            sum_spearman_rr_knockout = 0.0
            sum_spearman_ladder = 0.0

            sum_ndcg_spearman_rr = 0.0
            sum_ndcg_spearman_sr = 0.0
            sum_ndcg_spearman_de = 0.0
            sum_ndcg_spearman_weighted = 0.0
            sum_ndcg_spearman_rr_knockout = 0.0
            sum_ndcg_spearman_ladder = 0.0
                
            for i in range(iterations):
                if param_type == "rounds":
                    rounds_num = param
                    matches_num_val = None
                    finish_all_rounds_flag = False
                elif param_type == "matches":
                    rounds_num = None
                    matches_num_val = param
                    finish_all_rounds_flag = False
                else:
                    raise ValueError("param_type 必须为 'rounds', 'matches' 或 'finish_all_rounds'之一")
                
                player_data = PlayerData(win_matrix.shape[0])
                rr_initial, rr_ranked = robin_round(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
                player_data = PlayerData(win_matrix.shape[0])
                sr_initial, sr_ranked = swiss_round(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
                player_data = PlayerData(win_matrix.shape[0])
                de_initial, de_ranked = double_elimination_random(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
                player_data = PlayerData(win_matrix.shape[0])
                weighted_initial, weighted_ranked = weighted_round_robin(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
                player_data = PlayerData(win_matrix.shape[0])
                rr_knockout_initial, rr_knockout_ranked = rr_knockout(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
                player_data = PlayerData(win_matrix.shape[0])
                ladder_initial, ladder_ranked = ladder_tournament(win_matrix, player_data.get_copy(), rounds_num, matches_num_val, finish_all_rounds_flag)
                
                sum_spearman_rr += calculate_Spearman_coefficient(rr_initial, rr_ranked)
                sum_spearman_sr += calculate_Spearman_coefficient(sr_initial, sr_ranked)
                sum_spearman_de += calculate_Spearman_coefficient(de_initial, de_ranked)
                sum_spearman_weighted += calculate_Spearman_coefficient(weighted_initial, weighted_ranked)
                sum_spearman_rr_knockout += calculate_Spearman_coefficient(rr_knockout_initial, rr_knockout_ranked)
                sum_spearman_ladder += calculate_Spearman_coefficient(ladder_initial, ladder_ranked)

                sum_ndcg_spearman_rr += calculate_ndcg_Spearman_coefficient(rr_initial, rr_ranked)
                sum_ndcg_spearman_sr += calculate_ndcg_Spearman_coefficient(sr_initial, sr_ranked)
                sum_ndcg_spearman_de += calculate_ndcg_Spearman_coefficient(de_initial, de_ranked)
                sum_ndcg_spearman_weighted += calculate_ndcg_Spearman_coefficient(weighted_initial, weighted_ranked)
                sum_ndcg_spearman_rr_knockout += calculate_ndcg_Spearman_coefficient(rr_knockout_initial, rr_knockout_ranked)
                sum_ndcg_spearman_ladder += calculate_ndcg_Spearman_coefficient(ladder_initial, ladder_ranked)
                
                print(f"Match: {match_name}; Distribution: {distribution_type}; DataType: {datatype}; Param: {param}; Iteration: {i+1}/{iterations} finished.")
            
            # 计算平均值
            avg_spearman_rr = sum_spearman_rr / iterations
            avg_spearman_sr = sum_spearman_sr / iterations
            avg_spearman_de = sum_spearman_de / iterations
            avg_spearman_weighted = sum_spearman_weighted / iterations
            avg_spearman_rr_knockout = sum_spearman_rr_knockout / iterations
            avg_spearman_ladder = sum_spearman_ladder / iterations
            
            avg_ndcg_spearman_rr = sum_ndcg_spearman_rr / iterations
            avg_ndcg_spearman_sr = sum_ndcg_spearman_sr / iterations
            avg_ndcg_spearman_de = sum_ndcg_spearman_de / iterations
            avg_ndcg_spearman_weighted = sum_ndcg_spearman_weighted / iterations
            avg_ndcg_spearman_rr_knockout = sum_ndcg_spearman_rr_knockout / iterations
            avg_ndcg_spearman_ladder = sum_ndcg_spearman_ladder / iterations

            row = {
                "Match": match_name,
                "Distribution Type": distribution_type,
                "Data Type": datatype,
                "Parameter Type": param_type,
                "Parameter": param,
                "Finish all rounds": 'False',
                "Spearman Round Robin": avg_spearman_rr,
                "Spearman Swiss Round": avg_spearman_sr,
                "Spearman Double Elimination": avg_spearman_de,
                "Spearman Weighted Round Robin": avg_spearman_weighted,
                "Spearman Round Robin Knockout": avg_spearman_rr_knockout,
                "Spearman Ladder Tournament": avg_spearman_ladder,
                "NDCG Spearman Round Robin": avg_ndcg_spearman_rr,
                "NDCG Spearman Swiss Round": avg_ndcg_spearman_sr,
                "NDCG Spearman Double Elimination": avg_ndcg_spearman_de,
                "NDCG Spearman Weighted Round Robin": avg_ndcg_spearman_weighted,
                "NDCG Spearman Round Robin Knockout": avg_ndcg_spearman_rr_knockout,
                "NDCG Spearman Ladder Tournament": avg_ndcg_spearman_ladder
            }
            
            # 将该行结果追加到 CSV 文件
            temp_df = pd.DataFrame([row])
            # 如果文件不存在，则写入表头；否则追加写入且不写入表头
            temp_df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))

if __name__ == '__main__':
    win_matrix = standard_matrix(5, 5)
    iterations = 5
    match_name = "Go"
    distribution_type = "Normal"
    datatype = "Real Data"
    
    # 例如：以轮次数控制，测试 rounds_num 从 1 到 10
    param_type = "rounds"
    param_values = list(range(10, 11))
    save_path = os.getcwd() 
    
    tournament_correlation(match_name, distribution_type, win_matrix, iterations, datatype, param_type, param_values, save_path)
