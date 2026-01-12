import os
import torch

def save_checkpoint(state, checkpoint_dir, filename="state.pt"):
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    best_path = os.path.join(checkpoint_dir, "state.pt")
    torch.save(state, best_path)
    print(f"--- Saved model to {best_path} ---")

def load_checkpoint(checkpoint_dir, device, filename="state.pt"):
    return torch.load(f'{checkpoint_dir}/{filename}', map_location=device)