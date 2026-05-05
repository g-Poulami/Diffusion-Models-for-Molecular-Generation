import torch
from torch_geometric.datasets import QM9

# 1. Load real data using the recommended attribute access
dataset = QM9(root='./data/QM9')
real_properties = dataset.y[:, :11] # Accessing via recommended attribute

# 2. Your generated tensor
generated = torch.tensor([[22.4536, -64.0092, 131.3618, -49.2166, -55.9049, 
                           104.6712, -119.2691, 175.4058, 8.2323, -94.3636, -127.3591]])

# 3. Find the closest property profile
distances = torch.norm(real_properties - generated, dim=1)
closest_idx = torch.argmin(distances).item()
closest_molecule = dataset[closest_idx]

# 4. Display results
print(f"Closest real molecule index: {closest_idx}")

# Note: QM9 data objects contain 'z' (atomic numbers) and 'pos' (coordinates)
# but the SMILES string is usually stored in the original .csv or a separate attribute.
# To see the chemical structure, we can look at the atomic numbers:
print(f"Atomic numbers in molecule: {closest_molecule.z}")
print(f"Number of atoms: {closest_molecule.x.shape[0]}")