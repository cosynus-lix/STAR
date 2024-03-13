import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Define a custom color palette for the lines
color_palette = ['#FF5733', '#FFC300', '#33FF57', '#3399FF', '#A63B33']
# Define the font size for titles
title_fontsize = 20
legend_fontsize = 18

from scipy import stats
from scipy.ndimage.filters import uniform_filter1d

def read_results(exp, alg):
    success = []
    dist = []
    n_runs = 0
    # file = 'paper_exp/' + exp + '/' + alg + '/' + exp + '_' + alg + '_' + str(run) + '_2.csv'
    dir = 'paper_exp/' + exp + '/' + alg + '/'
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            frames = []
            s = []
            d = []
            n_runs += 1
            file = dir + file
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        frames.append(int(row[0]))
                        s.append(float(row[1]))
                        d.append(float(row[2]))
                        line_count += 1
                print(f'Processed {line_count} lines.')
            success.append(s)
            dist.append(d)
    success = np.array(success)
    dist = np.array(dist)
    return frames, np.mean(success, axis=0), np.std(success, axis=0)

def plot(exp, n_runs, ax, window=30, mode='nearest', label='exp'):
    if exp == 'AntMaze':
        alg = ['gara', 'garaold', 'hrac', 'hiro', 'lesson']
    else:
        alg = ['gara', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a)

    # Plot gara
    ax.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode),'tab:orange', label='StarGARA')
    ax.fill_between(frames, uniform_filter1d(r['gara'] + v['gara'] / 2, size=window, mode=mode),
                    uniform_filter1d(r['gara'] - v['gara'] / 2, size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    # Plot garaold
    if exp == 'AntMaze':
        ax.plot(frames, uniform_filter1d(r['garaold'], size=window, mode=mode), 'tab:red', label='GARA')
        ax.fill_between(frames, uniform_filter1d(r['garaold'] + v['garaold'] / 2, size=window, mode=mode),
                        uniform_filter1d(r['garaold'] - v['garaold'] / 2, size=window, mode=mode), facecolor='tab:red',
                        alpha=0.2)
    # Plot hrac
    ax.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    ax.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac'] / 2, size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac'] / 2, size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    ax.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    ax.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro'] / 2, size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro'] / 2, size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    ax.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    ax.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson'] / 2, size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson'] / 2, size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

    return ax

def plot_all(n_runs=3, window=30, mode='nearest'):
    # Create a new figure and specify the number of rows and columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    plt.subplots_adjust(wspace=0.4)  # Adjust the horizontal space between subplots

    for ax in axes:
        # Add a faint grid behind the graphs
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Initialize legend labels
    legend_labels = []

    alg_legend_labels = []  # List to store legend labels for algorithms

    experiments = ['AntMaze', 'AntFall', 'AntMazeCam']
    # Call plot function for each experiment and customize the colors
    for i, exp_name in enumerate(['AntMaze', 'AntFall', 'AntMazeCam']):
        ax = axes[i]
        label = exp_name
        ax = plot(exp_name, 3 , ax, window=30, mode='nearest', label=label)

        legend_labels.append(label)
        # Add axis labels to each subplot
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Average success rate")

        # Set the experiment name as the title for each graph
        ax.set_title(exp_name, fontsize=title_fontsize)

        # plt.legend(loc="lower right", bbox_to_anchor=(1., 1.02), borderaxespad=0., ncol=5)

        # Collect legend labels for algorithms from each subplot
        legend = ax.get_legend()
        if legend:
            alg_legend_labels.extend(legend.get_texts())

    # fig.legend(loc='lower center', bbox_to_anchor=(1., 1.02), borderaxespad=0., ncol=5)
    # alg_legend_labels = ['gara', 'garaold', 'hrac', 'hiro', 'lesson']
    # Create a custom legend for algorithms outside of subplots
    alg_legend = fig.legend(alg_legend_labels, loc="upper left", bbox_to_anchor=(0.13, 0.87), fontsize=legend_fontsize,
                            title='Algorithms')

    # Add the algorithm legend back to the figure
    fig.add_artist(alg_legend)
    plt.show()

