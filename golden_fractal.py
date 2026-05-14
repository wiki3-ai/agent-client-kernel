import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ipywidgets import interact, IntSlider

def draw_golden_spiral_fractal(ax, center_x, center_y, size, level, max_level):
    if level > max_level:
        return

    phi = (1 + np.sqrt(5)) / 2
    colors = plt.cm.viridis(np.linspace(0, 1, max_level + 1))
    color = colors[level]

    half_size = size / 2
    rect = patches.Rectangle(
        (center_x - half_size, center_y - half_size),
        size, size,
        linewidth=2 if level == 0 else 1,
        edgecolor=color,
        facecolor='none',
        alpha=0.8
    )
    ax.add_patch(rect)

    if level <= max_level:
        arc = patches.Arc(
            (center_x - half_size, center_y - half_size),
            size, size,
            theta1=0, theta2=90,
            linewidth=1.5,
            color=color
        )
        ax.add_patch(arc)

    if level < max_level:
        new_size = size / phi
        new_center_x = center_x + half_size - new_size/2
        new_center_y = center_y + half_size - new_size/2

        draw_golden_spiral_fractal(
            ax, new_center_x, new_center_y, 
            new_size, level + 1, max_level
        )

def plot_fractal(num_levels=5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    initial_size = 8

    draw_golden_spiral_fractal(ax, 0, 0, initial_size, 0, num_levels)

    margin = initial_size / 2 + 1
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Golden Ratio Fractal - {num_levels} Levels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_fractal, 
         num_levels=IntSlider(value=5, min=1, max=10, step=1, description='Levels:'))
