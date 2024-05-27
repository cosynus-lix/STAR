import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import os
import csv
title_fontsize = 15
legend_fontsize = 12

dir = '/results/partitions/'

def transform(regions):
    trans = []
    for r in regions:
        t = (r[0],r[3],r[1],r[4],r[2],r[5])
        trans.append(t)
    return trans

def plot_sphere(ax, position, radius):
    """Plot a sphere at the specified position with the given radius."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = position[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = position[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = position[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='yellow', alpha=1)

def calculate_cube_colors(data_values, cmap):
    # Normalize data values to the range [0, 1]
    normalized_values = (data_values - np.min(data_values)) / (np.max(data_values) - np.min(data_values))
    # normalized_values = (data_values - np.min([0])) / (np.max([5000]) - np.min([0]))
    # Apply the colormap to get colors
    face_colors = cmap(normalized_values)
    return face_colors, normalized_values

def plot_colored_cube(ax, bounds, alpha, cube_color, edgecolors = 'k'):
    """Plot a cube colored with a single color based on data values."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    vertices = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax)
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolors=edgecolors, facecolors=cube_color))

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_Ant_Fall(regions, data_values, ax, title):
    # Arrow initialization
    x_start = -10  # x-coordinate of the starting point
    y_start = -10  # y-coordinate of the starting point
    z_start = 4
    x_end = -10  # x-coordinate of the ending point
    y_end = -10
    z_end = 4

    fixed_blocks = [
        (-8, 16, 0, 12, 0, 4),
        (-8, 16, 16, 32, 0, 4),
    ]
    # Example usage
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for i in range(len(fixed_blocks)):
        # cube_bounds = (0, 4, 0, 5, 1, 3)  # (xmin, xmax, ymin, ymax, zmin, zmax)
        cube_bounds = fixed_blocks[i]
        plot_colored_cube(ax, cube_bounds, 0.2, 'grey', 'grey')
        # Set the experiment name as the title for each graph
        ax.set_title(title, fontsize=title_fontsize, y=-0.25)

    regions = transform(regions)

    cmap = plt.get_cmap('viridis')

    # Calculate the single color based on data values and colormap
    cube_colors, visits = calculate_cube_colors(data_values, cmap)

    for i in range(len(regions)):
        r = regions[i]
        if visits[i] > 0:
            plot_colored_cube(ax, r, 0.7, cube_colors[i], 'k')

        if x_start > -10 and y_start > -10:
            x_end = (r[2] - r[0])/2 + r[0]  # x-coordinate of the ending point
            y_end = (r[3] - r[1])/2 + r[1]  # y-coordinate of the ending point
            # Create an arrow patch
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                        arrowprops=dict(arrowstyle='->', color='r', linewidth=2))

        # Define arrow parameters (start and end points)
        x_start = (r[2] - r[0])/2 + r[0]  # x-coordinate of the starting point
        y_start = (r[3] - r[1])/2 + r[1]  # y-coordinate of the starting point


    movable_block = (8, 12, 8, 12, 4, 8)
    plot_colored_cube(ax, movable_block, 0.5, 'r', 'r')
    # Set axis limits
    ax.set_xlim(-12, 20)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 10)

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Define sphere parameters (position and radius)
    sphere_position = (0, 27, 6)  # Specify the position (x, y, z)
    sphere_radius = 1  # Specify the radius

    # plot_sphere(ax, sphere_position, sphere_radius)

    # plt.show()

def plot_Ant_Fall_seq(regions, visits):
    fig = plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
    titles = ['1M Timesteps', '2M Timesteps', '3M Timesteps']
    for i in range(len(regions)):
        # Calculate the left position for each subplot
        ax = fig.add_subplot(1, len(regions), i + 1, projection='3d')
        ax.view_init(elev=50, azim=-60)  # Set the elevation (vertical angle) and azimuth (horizontal angle)
        plot_Ant_Fall(regions[i], visits[i], ax, titles[i])

    cmap = plt.get_cmap('viridis')
    # Create a colorbar to show the mapping of data values to colors
    cax = fig.add_axes([0.96, 0.1, 0.01, 0.8])  # Position and size of the colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
    cbar.set_label('Frequency of visits (Normalized)')
    # Set a fixed angle for all subplots
    plt.subplots_adjust(left=0.03, right=0.95, wspace=0.1)  # Adjust subplot spacing
    plt.show()
    fname = "./antfall_repr.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')

def plot_Ant_Maze(regions, data_values, ax, title):
    cmap = plt.get_cmap('viridis')

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

    color, visits = calculate_cube_colors(data_values, cmap)
    for i in range(len(regions)):
        r = regions[i]
        rectangle = patches.Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1], alpha=0.7, facecolor=color[i], edgecolor='k')
        ax.add_patch(rectangle)
        ax.set_title(title, fontsize=title_fontsize, y=-0.25)

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

def plot_Ant_Maze_seq(regions, visits):
    fig = plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
    # Calculate the width and height for each subplot
    width = 1 / 3
    height = 1
    titles = ['1M Timesteps', '2M Timesteps', '3M Timesteps']

    for i in range(len(regions)):
        # Calculate the left position for each subplot
        left = i * width

        # ax = fig.add_axes([left, 0, width, height], projection='3d')
        ax = fig.add_subplot(1, len(regions), i + 1)
        plot_Ant_Maze(regions[i], visits[i], ax, titles[i])

    cmap = plt.get_cmap('viridis')
    # Create a colorbar to show the mapping of data values to colors
    cax = fig.add_axes([0.96, 0.1, 0.01, 0.8])  # Position and size of the colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
    cbar.set_label('Frequency of visits (Normalized)')
    # Set a fixed angle for all subplots
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)  # Adjust subplot spacing
    plt.show()
    fname = "./antmaze_repr.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')

def read_partitions():
    visits = {'AntMaze':[], 'AntFall':[]}
    regions = {'AntMaze':[], 'AntFall':[]}
    experiments = ['AntMaze', 'AntFall']
    for i in range(3):
        for e in experiments:
            file = dir + e + str(i) + "_BossPartitions.pth"
            v = []
            r = []
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        v = [int(rec) for rec in row]
                        line_count += 1
                    else:
                        r.append([float(rec) for rec in row])

            visits[e].append(v)
            regions[e].append(r)

    return visits, regions

v, r = read_partitions()
plot_Ant_Maze_seq(r['AntMaze'], v['AntMaze'])
plot_Ant_Fall_seq(r['AntFall'], v['AntFall'])