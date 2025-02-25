import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import os
from scipy.stats import norm
from torch.distributions.normal import Normal
import torch.nn.functional as F
from Matrix_Process.Tennis_rating import tennis_rating
from Matrix_Process.Go_rating import go_rating
from Matrix_Process.StarCraft_rating import scraft_rating
from Matrix_Process.Badminton_rating import badminton_rating

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 保证每次结果一致（但可能会牺牲一定的性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GaussianMixtureModel(nn.Module):
    def __init__(self, k):
        super(GaussianMixtureModel, self).__init__()
        self.k = k
        self.A = nn.Parameter(torch.randn(k))  # Means of the Gaussian components
        self.B = nn.Parameter(torch.randn(k).abs())  # Standard deviations (ensuring positivity)

    def forward(self, x):
        """
        Expects x to be a tensor of shape [2]. Here, we combine the two elements
        (for example, by subtracting them) to obtain a scalar input.
        """
        # Combine the two elements into a single scalar input
        x_scalar = x[0] - x[1]

        # Compute the normalization denominator (total weight)
        total_weight = sum(1 / (2 * n - 1) for n in range(1, self.k + 1))
        mixture_sum = 0

        for i in range(self.k):
            weight = 1 / (2 * (i + 1) - 1)
            # Here we use B[i] directly as standard deviation.
            gaussian = Normal(self.A[i], F.softplus(self.B[i]))
            # Evaluate the probability density at x_scalar and weight it.
            mixture_sum += weight * torch.exp(gaussian.log_prob(x_scalar))

        return mixture_sum / total_weight


def paremeter_training(matrix, match_name, k):
    # Generate the true validation matrix (32x32) from your tennis_rating function.
    true_R = torch.tensor(matrix, dtype=torch.float32)  # Convert to a PyTorch tensor

    model = GaussianMixtureModel(k=k)  # Initialize the model with 3 Gaussian components
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # Mean Squared Error Loss

    n = true_R.size(0)  # matrix size (32)
    num_epochs = 1000

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Build the predicted winning rate matrix R_pred (upper triangular)
        R_pred = torch.zeros_like(true_R)
        for i in range(n):
            for j in range(i + 1, n):
                # Create a two-element tensor as input (e.g., [i, j])
                input_tensor = torch.tensor((float(i), float(j)), dtype=torch.float32)
                R_pred[i, j] = model(input_tensor)

        # Create a mask to select only the upper-triangular entries (i < j)
        mask = torch.triu(torch.ones_like(true_R), diagonal=1)
        loss = criterion(R_pred[mask == 1], true_R[mask == 1])

        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    # After training, print the final loss and the predicted matrix.
    print("\nFinal Loss:", loss.item())
    print("Final Predicted Upper-Triangular Matrix (R_pred):")
    # print(R_pred)
    print(model.A.detach().numpy())
    print(model.B.detach().numpy())
    # Extract the parameters A and B, and save them to a CSV file "GMM.csv"
    params_df = pd.DataFrame({
        'A': model.A.detach().numpy(),
        'B': model.B.detach().numpy()
    })
    folder_path = f"best_parameters/{k}_matrix_{matrix.shape}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    params_df.to_csv(f"best_parameters/{k}_matrix_{matrix.shape}/{match_name}_GMM.csv", index=False)
    print(f"\nParameters saved to {k}_matrix_{matrix.shape}/{match_name}/GMM.csv")


def get_mixture_pdf(file_name):
    """
    读取 CSV 文件中的 GMM 参数，并计算混合分布的概率密度。

    返回：
        x: 实际实力值范围（未归一化）
        mixture_pdf: 对应的混合分布概率密度（未归一化）
        x_min, x_max: x 的最小和最大值，用于归一化
    """
    df = pd.read_csv(file_name)
    A = df['A'].values  # 各分量均值
    B = df['B'].values  # 各分量标准差
    B_transformed = np.log1p(np.exp(B))
    # 定义权重： weight = 1 / (2*(i+1) - 1)
    n_components = len(A)
    weights = np.array([1 / (2 * (i + 1) - 1) for i in range(n_components)])
    normalized_weights = weights / np.sum(weights)

    # 根据均值和标准差设置 x 轴范围
    x_min = np.min(A) - 4 * np.max(B)
    x_max = np.max(A) + 4 * np.max(B)
    x = np.linspace(x_min, x_max, 1000)

    # 计算加权混合分布的概率密度
    mixture_pdf = np.zeros_like(x)
    for i in range(n_components):
        mixture_pdf += normalized_weights[i] * norm.pdf(x, loc=A[i], scale=B_transformed[i])

    return x, mixture_pdf, x_min, x_max
def plot_GMMs(folder):
    """
    读取指定文件夹下的四个 GMM 参数文件，并绘制两幅图：
    1. 一幅图中包含四条未归一化的混合分布曲线（横坐标为实际实力值）。
    2. 一幅图中包含四条归一化的混合分布曲线（横纵坐标均映射到 [0,1]）。
    """
    # 构造文件名列表（注意 CSV 文件名应与此处保持一致）
    files = [
        folder + "GO_GMM.csv",
        folder + "StarCraft_GMM.csv",
        folder + "Tennis_GMM.csv",
        folder + "Badminton_GMM.csv"
    ]
    titles = ["GO", "StarCraft", "Tennis", "Badminton"]

    # -------------------------------
    # 图1：未归一化的混合分布，将四条曲线绘制在同一幅图中
    plt.figure(figsize=(10, 6))
    for file, title in zip(files, titles):
        x, mixture_pdf, x_min, x_max = get_mixture_pdf(file)
        plt.plot(x, mixture_pdf, label=title)
    plt.title("Combined GMM Models (Unnormalized)")
    plt.xlabel("Skill Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

    # -------------------------------
    # 图2：归一化后的混合分布，将四条曲线绘制在同一幅图中
    plt.figure(figsize=(10, 6))
    for file, title in zip(files, titles):
        x, mixture_pdf, x_min, x_max = get_mixture_pdf(file)
        # 对 x 轴进行线性归一化到 [0,1]
        x_normalized = (x - x_min) / (x_max - x_min)
        # 对概率密度也进行 min-max 归一化到 [0,1]
        mixture_pdf_normalized = (mixture_pdf - mixture_pdf.min()) / (mixture_pdf.max() - mixture_pdf.min())
        plt.plot(x_normalized, mixture_pdf_normalized, label=title)
    plt.title("GMM Models Strength Distribution (Normalized)")
    plt.xlabel("Normalized Strength Value")
    plt.ylabel("Normalized Probability Density")
    plt.legend()
    plt.show()
if __name__ == '__main__':

    # set_random_seed(42)
    #
    # matches = ['StarCraft', 'Tennis', 'Go', 'Badminton']
    # player_number = 32
    # a = 1
    # for k in [3, 5, 7, 9, 11]:
    #     for match in matches:
    #         if match == 'StarCraft':
    #             matrix = scraft_rating(player_number, a, False)
    #         if match == 'Tennis':
    #             matrix = tennis_rating(player_number, a, False)
    #         if match == 'Go':
    #             matrix = go_rating(player_number, a, False)
    #         if match == 'Badminton':
    #             matrix = badminton_rating(player_number, a, False)
    #         paremeter_training(matrix, match_name = match, k = k)

    plot_GMMs("best_parameters/11_matrix_33/")