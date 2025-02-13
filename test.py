import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from Matrix_Process.Tennis_rating import tennis_rating


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
            gaussian = Normal(self.A[i], self.B[i])
            # Evaluate the probability density at x_scalar and weight it.
            mixture_sum += weight * torch.exp(gaussian.log_prob(x_scalar))

        return mixture_sum / total_weight


def paremeter_training(matrix):
    # Generate the true validation matrix (32x32) from your tennis_rating function.
    true_R = torch.tensor(matrix, dtype=torch.float32)  # Convert to a PyTorch tensor

    model = GaussianMixtureModel(k=3)  # Initialize the model with 3 Gaussian components
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # Mean Squared Error Loss

    n = true_R.size(0)  # matrix size (32)
    num_epochs = 100

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
    params_df.to_csv("GMM.csv", index=False)
    print("\nParameters saved to GMM.csv")

if __name__ == '__main__':
    matrix = tennis_rating(32, 2, False)

    paremeter_training(matrix)
