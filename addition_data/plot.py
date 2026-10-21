import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_activations(feature_id, values, indices, data_range):
    """
    Plots a 2D heatmap of SAE feature activations for p1 + p2 combinations.
    Fixed to ensure grid lines cleanly bound the data squares.
    """
    min_val, max_val = data_range
    N = max_val - min_val + 1 
    
    # Create an empty grid for the heatmap
    activation_grid = np.full((N, N), np.nan)
    
    # 1. Create boolean mask
    mask = (indices == feature_id) & (values > 1e-3)
    b_indices, d_indices = torch.where(mask)
    valid_acts = values[b_indices, d_indices]
    
    # 2. Map linear index back to 2D coordinates
    p1_coords = b_indices // N
    p2_coords = b_indices % N
    
    # 3. Populate the grid
    activation_grid[p2_coords.cpu().numpy(), p1_coords.cpu().numpy()] = valid_acts.cpu().numpy()
    
    # ==========================================
    # Plotting
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    
    cmap = plt.cm.Oranges
    cmap.set_bad(color='white') 
    
    cax = ax.imshow(
        activation_grid, 
        origin='lower', 
        cmap=cmap, 
        # Extent centers the integer values perfectly inside the squares
        extent=[min_val - 0.5, max_val + 0.5, min_val - 0.5, max_val + 0.5]
    )
    
    # 4. Determine tick spacing based on range size
    if N <= 15:
        tick_step = 1
    elif N <= 50:
        tick_step = 5
    elif N <= 100:
        tick_step = 10
    else:
        tick_step = max(10, N // 10)
        
    # Major ticks for the numerical labels (centered on the squares)
    major_ticks = np.arange(min_val, max_val + 1, tick_step)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    
    # Minor ticks for the grid lines (shifted by 0.5 to border the squares)
    minor_ticks = np.arange(min_val - 0.5, max_val + 1.5, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    
    # 5. Draw the grid ONLY on the minor ticks
    grid_alpha = 0.7 if N >= 20 else 0.4
    grid_linewidth = 1.5 if N >= 20 else 0.8
    
    ax.grid(True, which='minor', color='#E0E0E0', linestyle='-', linewidth=grid_linewidth, alpha=grid_alpha)
    ax.grid(False, which='major') # Turn off the major grid lines
    
    # Remove minor tick marks (the little lines on the axes) while keeping the grid
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Stylize the plot
    ax.set_xlabel('p1', fontsize=12)
    ax.set_ylabel('p2', fontsize=12)
    ax.set_title(f'SAE Feature {feature_id} Activations', fontsize=14, pad=15)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Strength', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()