from rdkit import Chem
from rdkit.Chem import Draw

# Use the SMILES from your previous step
smiles_methane = "C" 
mol = Chem.MolFromSmiles(smiles_methane)

# Save the molecule image to your folder
Draw.MolToFile(mol, 'generated_molecule.png')