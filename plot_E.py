import numpy as np
import os
import matplotlib.pyplot as plt
from validation import E_vector_calculate
# from validation import E_vector_calculate_ndcg
from Go_rating import go_rating
from Scraft_rating import scraft_rating
from Tennis_rating import tennis_rating
from Badminton_rating import badminton_rating
from validation import best_theta_simulation

def plot_E_matches(E_list, matches):
    """
    Plots the E vector trends for multiple matches on the same graph.

    Parameters:
        E_list (list of lists): A list containing E vectors for each match.
        matches (list of str): A list of match names corresponding to E_list.
    """
    plt.figure(figsize=(10, 6))

    # Colors for each match
    colors = ['blue', 'green', 'red', 'purple']

    # Plot each match's E vector
    for i, (E, match_name) in enumerate(zip(E_list, matches)):
        x = np.arange(1, len(E) + 1)  # x轴从1开始
        plt.plot(x, E, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'{match_name} R Trend')

    # 设置x轴刻度
    max_length = max(len(E) for E in E_list)
    plt.xticks(np.arange(1, max_length + 1))  # 统一x轴刻度范围

    # 设置图表属性
    plt.title('R Vector Trends for Multiple Matches', fontsize=14)
    plt.xlabel('Rank 1 to Rank 33', fontsize=12)
    plt.ylabel('R Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# def plot_E():
#     # matches = ['Go', 'Tennis', 'Star','Badminton']
#     matches = ['Go', 'Tennis', 'Star']
#     E_list = []

#     # Calculate E vectors for each match
#     for match_name in matches:
#         if match_name == 'Go':
#             winning_matrix = go_rating()
#         elif match_name == 'Tennis':
#             winning_matrix = tennis_rating()
#         elif match_name == 'Badminton':
#             winning_matrix = badminton_rating()
#         elif match_name == 'Star':
#             winning_matrix = scraft_rating()

#         E = E_vector_calculate(winning_matrix)
#         E_list.append(E[:-1])

#     # Plot all E trends
#     plot_E_matches(E_list, matches)



def plot_R_and_SR(match_name,best_uniform, best_powerlaw, best_normal, E, u_theta, PL_theta, normal_theta, save_figure_path):
    """
    Plots the four lists (best_uniform, best_powerlaw, best_normal, E) on the same graph,
    following the single-line plot style.

    Parameters:
        best_uniform (list or np.ndarray): List of best uniform values.
        best_powerlaw (list or np.ndarray): List of best power law values.
        best_normal (list or np.ndarray): List of best normal values.
        E (list or np.ndarray): E vector.
    """
    plt.figure(figsize=(10, 6))

    # X-axis indices for each list
    x_uniform = np.arange(1, len(best_uniform))
    x_powerlaw = np.arange(1, len(best_powerlaw))
    x_normal = np.arange(1, len(best_normal))
    x_E = np.arange(1, len(E))

    plot_num = len(E)-1

    # Plot each list with single-line style
    plt.plot(x_uniform, best_uniform[:plot_num], marker='o', linestyle='-', color='blue', label=f'Best Uniform theta = {u_theta}')
    plt.plot(x_powerlaw, best_powerlaw[:plot_num], marker='o', linestyle='-', color='green', label=f'Best Power Law theta = {PL_theta}')
    plt.plot(x_normal, best_normal[:plot_num], marker='o', linestyle='-', color='red', label=f'Best Normal theta = {normal_theta}')
    plt.plot(x_E, E[:plot_num], marker='o', linestyle='-', color='purple', label='R Vector')

    # Setting x-axis ticks
    max_length = max(len(best_uniform), len(best_powerlaw), len(best_normal), len(E))
    plt.xticks(np.arange(1, max_length + 1))

    # Add titles, labels, and legend
    plt.title(f'{match_name} Match -- Trends of R and Simulated R', fontsize=14)
    plt.xlabel('Index (starting from 1)', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_figure_path, f'{match_name}_R_SR.png'))

def plot_single_match(match_name,u_theta, PL_theta, normal_theta, match_to_matrix, save_figure_path, num_players):
    winning_matrix = []
    winning_matrix = match_to_matrix[match_name]
    best_uniform = best_theta_simulation(num_players, distribution_type='Uniform', theta=u_theta)
    best_powerlaw = best_theta_simulation(num_players, distribution_type='PL', theta=PL_theta)
    best_normal = best_theta_simulation(num_players, distribution_type='Normal', theta=normal_theta)
    E = E_vector_calculate(winning_matrix, num_players)
    plot_R_and_SR(match_name,best_uniform, best_powerlaw, best_normal, E, u_theta, PL_theta, normal_theta, save_figure_path)

if __name__ == '__main__':
    match = 'Go'
    plot_single_match(match,1,1,1)