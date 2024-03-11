import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import numpy as np

def transform(regions):
    trans = []
    for r in regions:
        t = (r[0],r[3],r[1],r[4],r[2],r[5])
        trans.append(t)
    return trans

def plot_Ant_Maze(regions):
    fig, ax = plt.subplots()

    # Arrow initialization
    x_start = -10  # x-coordinate of the starting point
    y_start = -10  # y-coordinate of the starting point
    x_end = -10  # x-coordinate of the ending point
    y_end = -10

    # Define rectangle parameters (position and size)
    rectangle = patches.Rectangle((-8, -8), 32, 4, linewidth=2, edgecolor='grey', facecolor='grey')
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((-8, 20), 32, 4, linewidth=2, edgecolor='grey', facecolor='grey')
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((-8, -4), 4, 24, linewidth=2, edgecolor='grey', facecolor='grey')
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((20, -4), 4, 24, linewidth=2, edgecolor='grey', facecolor='grey')
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((-4, 8), 20, 8, linewidth=2, edgecolor='grey', facecolor='grey')
    ax.add_patch(rectangle)

    for r in regions:
        rectangle = patches.Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1] , alpha=0.7, facecolor='g', edgecolor='k')
        ax.add_patch(rectangle)
        if x_start > -10 and y_start > -10:
            x_end = (r[2] - r[0])/2 + r[0]  # x-coordinate of the ending point
            y_end = (r[3] - r[1])/2 + r[1]  # y-coordinate of the ending point
            # Create an arrow patch
            arrow = patches.FancyArrow(x_start, y_start, x_end - x_start, y_end - y_start,
                                       width=0.1, head_width=0.6, head_length=0.6, edgecolor='b', facecolor='b')
            ax.add_patch(arrow)

        # Define arrow parameters (start and end points)
        x_start = (r[2] - r[0])/2 + r[0]  # x-coordinate of the starting point
        y_start = (r[3] - r[1])/2 + r[1]  # y-coordinate of the starting point

    # Define circle parameters (position and radius)
    x = 0  # x-coordinate of the center
    y = 16  # y-coordinate of the center
    radius = 0.5  # Radius of the circle

    # Create a Circle patch
    circle = patches.Circle((x, y), radius, linewidth=2, edgecolor='y', facecolor='y')

    # Add the circle to the axis
    ax.add_patch(circle)

    # Set axis limits (optional)
    ax.set_xlim(-8, 24)
    ax.set_ylim(-8, 24)

    # Show the plot
    plt.show()

# Example set of Regions
regions_antmaze = [
    (0.0,0.0,4.0,8.0),
    (4.0,0,8,8),
    (8,5,20,8),
    (15,8,20,20),
    (8,15,15,20),
    (0,15,8,20),
]

# plot_Ant_Maze(regions_antmaze)