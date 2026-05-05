import torch
from models.diffusion import ConditionalDiffusion

# Load your trained model
model = ConditionalDiffusion(input_dim=11, cond_dim=1)
# model.load_state_dict(torch.load('model_weights.pth')) # If you saved weights

def sample(target_property_value):
    model.eval()
    with torch.no_grad():
        # Start with pure Gaussian noise (Reverse Process)
        x = torch.randn(1, 11) 
        y = torch.tensor([[target_property_value]]) # The "Condition"
        
        # Iteratively denoise
        for t in reversed(range(1000)):
            t_batch = torch.full((1,), t)
            predicted_noise = model(x, t_batch, y)
            
            # Simplified DDPM update step
            # x = (x - predicted_noise * beta_factor) ... 
            # (In a full implementation, use the DDPM posterior mean formula)
            x -= 0.01 * predicted_noise 
            
    return x

# Generate a molecule property set with a target Dipole Moment of 1.5
generated_properties = sample(1.5)
print(f"Generated Molecular Properties: {generated_properties}")