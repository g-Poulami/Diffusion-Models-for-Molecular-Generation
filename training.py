import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from models.diffusion import ConditionalDiffusion

# 1. Setup Device and Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Downloads QM9 to your data folder
dataset = QM9(root='./data/QM9')

# 2. Diffusion Hyperparameters (DDPM Schedule)
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 3. Initialize Model, Optimizer, and Loss Tracker
# input_dim=11 for QM9 properties; cond_dim=1 for target property
model = ConditionalDiffusion(input_dim=11, cond_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = []

def train(epochs=5):
    print(f"Starting training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        # Mini-project: training on a representative subset
        for batch in dataset[:2000]:
            # x0: Real molecular properties[cite: 1]
            x0 = batch.y[:, :11].to(device) 
            # y: Conditioning property (e.g., alpha polarizability)[cite: 1]
            y = batch.y[:, 0].unsqueeze(-1).to(device) 
            
            # Sample random timesteps
            t = torch.randint(0, T, (x0.shape[0],)).to(device)
            noise = torch.randn_like(x0)
            
            # Forward Diffusion: Add noise to data[cite: 1]
            a_cp = alphas_cumprod[t].unsqueeze(-1)
            xt = torch.sqrt(a_cp) * x0 + torch.sqrt(1 - a_cp) * noise
            
            # Reverse Diffusion: Predict the added noise[cite: 1]
            predicted_noise = model(xt, t, y)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / 2000
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    # Save the model weights for sampling later[cite: 2]
    torch.save(model.state_dict(), 'diffusion_model.pth')
    print("Model saved as diffusion_model.pth")

# 4. Function to generate the required Training Curve[cite: 1]
def save_training_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o', linestyle='-', color='b')
    plt.title('Conditional Diffusion Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('training_curve.png') # Generates the deliverable image[cite: 1]
    print("Training curve saved as training_curve.png")

if __name__ == "__main__":
    train(epochs=10)
    save_training_plot()