import numpy as np
import os
import matplotlib.pyplot as plt

from Go_rating import go_rating
from Tennis_rating import tennis_rating
from Badminton_rating import badminton_rating
from validation import simulation

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

    Uniform = read_file(f'Result/{filename}/Uniform.txt')
    PL = read_file(f'Result/{filename}/PL.txt')
    Normal = read_file(f'Result/{filename}/Normal.txt')

    plt.figure(figsize=(10, 6))

    plt.plot(theta_values, Uniform, linestyle='--', color='r', label='Uniform')
    plt.plot(theta_values, PL, linestyle='--', color='blue', label='PowerLaw')
    plt.plot(theta_values, Normal, linestyle='--', color='gray', label='Normal')

    plt.xlabel('theta')
    plt.ylabel('Difference(D)')
    plt.title(f'Comparison of Different Distributions in {filename} match')

    plt.legend()

    plt.grid(True)
    plt.show()
    return 0

if __name__ == "__main__":

    winning_matrix = []

    match_name = 'Tennis'
    if match_name == 'Go':
        winning_matrix = go_rating()
    if match_name == 'Tennis':
        winning_matrix = tennis_rating()
    if match_name == 'Badminton':
        winning_matrix = badminton_rating()
    # strength_type = 'Uniform'
    # strength_type = 'PL'
    # strength_type = 'Normal'
    strength = ['Normal','Uniform','PL']
    all_D = []
    for strength_type in strength:
        for iteration in range(100):
            # D, min_theta , D_min = simulation(winning_matrix,distribution_type=strength_type)#三种分布方式 'Uniform','PL','Normal' 分别代表均匀分布，幂律分布，正态分布。
            D, _, _ = simulation(winning_matrix,distribution_type=strength_type)  # 三种分布方式 'Uniform','PL','Normal' 分别代表均匀分布，幂律分布，正态分布。
            all_D.append(D)
        all_D_array = np.array(all_D)
        D_mean = np.mean(all_D_array, axis=0)
        save_result_to_txt(D_mean,f'Result/{match_name}/{strength_type}.txt')
    plot_function(match_name)