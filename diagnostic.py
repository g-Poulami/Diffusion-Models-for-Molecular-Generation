import torch
import matplotlib.pyplot as plt
import numpy as np
from models.diffusion import ConditionalDiffusion

# 1. Initialization and Model Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConditionalDiffusion(input_dim=11, cond_dim=1).to(device)

# Load the weights generated from your training.py
try:
    model.load_state_dict(torch.load('diffusion_model.pth', map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("Error: diffusion_model.pth not found. Run training.py first.")

# 2. Function: Denoising Progression (Visualizing the "Reverse" Process)
def save_denoising_progression(target_val=1.5):
    steps_to_save = [1000, 750, 500, 250, 0]
    trajectories = []
    
    with torch.no_grad():
        # Start from pure Gaussian noise
        x = torch.randn(1, 11).to(device)
        y = torch.tensor([[target_val]]).to(device)
        
        for t in reversed(range(1000)):
            t_batch = torch.full((1,), t).to(device)
            # Predict noise based on current state and target property
            pred_noise = model(x, t_batch, y)
            
            # Update step (simplified DDPM reverse step)
            x -= 0.01 * pred_noise 
            
            if t in steps_to_save:
                trajectories.append(x.cpu().squeeze().numpy())

    # Create Heatmap of Property Development[cite: 1]
    plt.figure(figsize=(10, 6))
    plt.imshow(trajectories, aspect='auto', cmap='magma')
    plt.colorbar(label='Property Value Intensity')
    plt.title(f'Reverse Diffusion Progression (Target Property: {target_val})')
    plt.yticks(range(len(steps_to_save)), steps_to_save)
    plt.ylabel('Timestep (t)')
    plt.xlabel('Molecular Descriptor Index (QM9 Features)')
    plt.savefig('denoising_progression.png')
    print("Saved: denoising_progression.png")

# 3. Function: Conditional vs. Unconditional Distribution[cite: 1]
def compare_distributions():
    # Generating samples for two different biological target profiles[cite: 1, 2]
    low_target_samples = []
    high_target_samples = []
    
    with torch.no_grad():
        for _ in range(50):
            # Sample 1: Target property = 0.5 (e.g., low polarizability)
            x_low = torch.randn(1, 11).to(device)
            y_low = torch.tensor([[0.5]]).to(device)
            
            # Sample 2: Target property = 2.5 (e.g., high polarizability)
            x_high = torch.randn(1, 11).to(device)
            y_high = torch.tensor([[2.5]]).to(device)
            
            # Quick 100-step sampling for diagnostic speed
            for t in reversed(range(0, 1000, 10)):
                t_b = torch.full((1,), t).to(device)
                x_low -= 0.1 * model(x_low, t_b, y_low)
                x_high -= 0.1 * model(x_high, t_b, y_high)
                
            low_target_samples.append(x_low.cpu().mean().item())
            high_target_samples.append(x_high.cpu().mean().item())

    # Plot Histograms to show the model "shifts" based on condition[cite: 1]
    plt.figure(figsize=(8, 5))
    plt.hist(low_target_samples, alpha=0.5, label='Target: 0.5', color='blue')
    plt.hist(high_target_samples, alpha=0.5, label='Target: 2.5', color='red')
    plt.title('Property-Guided Generation: Output Distribution Shift')
    plt.xlabel('Mean Generated Property Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('conditional_comparison.png')
    print("Saved: conditional_comparison.png")

if __name__ == "__main__":
    save_denoising_progression(target_val=1.5)
    compare_distributions()