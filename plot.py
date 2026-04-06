import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming you have a dataframe 'df' with columns:
# - 'time_cycles': the x-axis values
# - 14 feature columns (already min-max normalized)

def plot_stacked_features(df, unit_nr, time_col='time_cycles', x_lim = [1, 400], vertical_lines=[20, 50]):
    """
    Create separate time series plots for each feature in individual windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time_cycles and normalized features (0-1 range)
    unit_nr : str
    time_col : str
        Name of the time column (default: 'time_cycles')
    vertical_lines : list
        X-axis positions for vertical dashed lines (default: [20, 50])
    
    Returns:
    --------
    figures : list
        List of figure objects, one for each feature
    axes : list
        List of axis objects, one for each feature
    """
    
    # Get feature columns (all columns except time_col)
    feature_cols = [col for col in df.columns if col != time_col]
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    figures = []
    axes = []
    
    # Create a separate plot for each feature
    for i, feature in enumerate(feature_cols):
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Use modulo to cycle through colors
        color = colors[i % len(colors)]
        
        # Plot the feature
        ax.plot(df[time_col], df[feature], color=color, linewidth=1.5, alpha=0.8)
        
        # Add vertical dashed lines
        for x_pos in vertical_lines:
            ax.axvline(x=x_pos, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        # Set labels and formatting
        ax.set_xlabel('Time Cycles', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{feature} - unit_nr: {unit_nr}', fontsize=14, fontweight='bold')
        
        # Set y-axis limits with some padding
        ax.set_ylim(-0.1, 1.1)
        
        # Set x-axis limits
        ax.set_xlim(x_lim[0], x_lim[1])
        
        # Add grid for better readability
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        figures.append(fig)
        axes.append(ax)
    
    return figures, axes


def plot_all_units_overlay_colorbar(df, unit_col='unit_nr', time_col='time_cycles', 
                                    feature_cols=None, x_lim = [1, 400], vertical_lines=[20, 50]):
    """
    Create separate time series plots for each feature showing all engine units overlaid.
    Uses a colorbar to show the unit mapping instead of a legend.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with unit, time_cycles, and normalized features (0-1 range)
    unit_col : str
        Name of the unit identifier column (default: 'unit_nr')
    time_col : str
        Name of the time column (default: 'time_cycles')
    feature_cols : list or None
        List of feature column names to plot. If None, uses all columns except unit_col and time_col
    vertical_lines : list
        X-axis positions for vertical dashed lines (default: [20, 50])
    
    Returns:
    --------
    figures : list
        List of figure objects, one for each feature
    axes : list
        List of axis objects, one for each feature
    """
    
    # Get feature columns if not specified
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [unit_col, time_col]]
    
    # Get unique units
    units = sorted(df[unit_col].unique())
    
    # Create color spectrum for units
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(units), vmax=max(units))
    
    figures = []
    axes = []
    
    # Create a separate plot for each feature
    for feat_idx, feature in enumerate(feature_cols):
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot each unit for this feature
        for unit in units:
            unit_data = df[df[unit_col] == unit].sort_values(time_col)
            color = cmap(norm(unit))
            
            ax.plot(unit_data[time_col], unit_data[feature], 
                   color=color, linewidth=0.8, alpha=0.6)
        
        # Add vertical dashed lines
        for x_pos in vertical_lines:
            ax.axvline(x=x_pos, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        # Set labels and formatting
        ax.set_xlabel('Time Cycles', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
        
        # Set y-axis limits with some padding
        ax.set_ylim(-0.1, 1.1)
        
        # Set x-axis limits
        ax.set_xlim(x_lim[0], x_lim[1])
        
        # Add grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Unit ID')
        
        # Adjust layout
        plt.tight_layout()
        
        figures.append(fig)
        axes.append(ax)
    
    return figures, axes