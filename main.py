import numpy as np
import os
import matplotlib.pyplot as plt
import random

from sympy import false

from Go_rating import go_rating
from Tennis_rating import tennis_rating
from Badminton_rating import badminton_rating
from Scraft_rating import  scraft_rating
from validation import simulation
from plot_E import plot_single_match

def save_result_to_txt(data_list,save_path):
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(save_path, "w", encoding="utf-8") as file:
        for item in data_list:
            file.write(f"{item}\n")
    return 0

def read_file(filename):
# Open the file in read mode ('r')
    with open(filename, 'r') as file:
        # Read each line, strip newline characters, and convert to the appropriate type
        number_list = [float(line.strip()) for line in file]
    return number_list

def plot_function(filename):
    theta_values = np.linspace(0, 2, 200)

    # Read data
    Uniform = read_file(f'Result/{filename}/Uniform.txt')
    PL = read_file(f'Result/{filename}/PL.txt')
    Normal = read_file(f'Result/{filename}/Normal.txt')

    plt.figure(figsize=(10, 6))

    # Plot distributions
    plt.plot(theta_values, Uniform, linestyle='--', color='r', label='Uniform')
    plt.plot(theta_values, PL, linestyle='--', color='blue', label='PowerLaw')
    plt.plot(theta_values, Normal, linestyle='--', color='gray', label='Normal')

    # Find and mark minimum values
    def mark_minima(data, color, label):
        min_value = np.min(data)
        min_indices = np.where(data == min_value)[0]
        theta_min = theta_values[min_indices[0]]
        for idx in min_indices:
            plt.axvline(x=theta_values[idx], color=color, linestyle=':', alpha=0.7)
            plt.scatter(theta_values[idx], min_value, color=color, label=f'theta={theta_min:.2f}' if idx == min_indices[0] else "")
        return theta_min
    Uniform_min = mark_minima(Uniform, 'r', 'Uniform')
    PL_min = mark_minima(PL, 'blue', 'PowerLaw')
    Normal_min = mark_minima(Normal, 'gray', 'Normal')

    # Add labels and legend
    plt.xlabel('theta')
    plt.ylabel('Difference(D)')
    plt.title(f'Comparison of Different Distributions in {filename} match')
    plt.legend()

    plt.grid(True)
    plt.show()
    return round(Uniform_min, 2), round(PL_min, 2), round(Normal_min, 2)

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

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    winning_matrix = []
    match_set = ['Go','Tennis','Badminton','Star']
    for match in match_set:
        match_name = match
    # match_name = 'Go'
        if match_name == 'Go':
            winning_matrix = go_rating()
        if match_name == 'Tennis':
            winning_matrix = tennis_rating()
        if match_name == 'Badminton':
            winning_matrix = badminton_rating()
        if match_name == 'Star':
            winning_matrix = scraft_rating()
        if match_name == 'Random':
            winning_matrix = standard_matrix(rows=33, cols=33)
        # strength_type = 'Uniform'
        # strength_type = 'PL'
        # strength_type = 'Normal'
        strength = ['Normal','Uniform','PL']
        # strength = ['Standard']
        all_D = []
        for strength_type in strength:
            for iteration in range(100):
                # D, min_theta , D_min = simulation(winning_matrix,distribution_type=strength_type)#三种分布方式 'Uniform','PL','Normal' 分别代表均匀分布，幂律分布，正态分布。
                D, _, _ = simulation(winning_matrix,distribution_type=strength_type,simulate_and_1_flag=false)  # 三种分布方式 'Uniform','PL','Normal' 分别代表均匀分布，幂律分布，正态分布。
                all_D.append(D)
            all_D_array = np.array(all_D)
            D_mean = np.mean(all_D_array, axis=0)
            save_result_to_txt(D_mean,f'Result_32/{match_name}/{strength_type}.txt')
            # save_result_to_txt(D_mean, f'Result/{match_name}/{strength_type}.txt')
        uniform_min, PL_min, Normal_min = plot_function(match_name)
        print(uniform_min, PL_min, Normal_min)
        plot_single_match(match_name, uniform_min, PL_min, Normal_min)
