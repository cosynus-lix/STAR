import numpy as np
import matplotlib.pyplot as plt
import csv

# Define a custom color palette for the lines
color_palette = ['#FF5733', '#FFC300', '#33FF57', '#3399FF', '#A63B33']
# Define the font size for titles
title_fontsize = 20
legend_fontsize = 18

from scipy import stats
from scipy.ndimage.filters import uniform_filter1d

def read_results(exp, alg, n_runs):
    success = []
    dist = []
    for run in range(n_runs):
        frames = []
        s = []
        file = 'paper_exp/' + exp + '/' + alg + '/' + exp + '_' + alg + '_' + str(run) + '_2.csv'
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
                    dist.append(float(row[2]))
                    line_count += 1
            print(f'Processed {line_count} lines.')
        success.append(s)
    success = np.array(success)
    return frames, np.mean(success, axis=0), np.std(success, axis=0)

def plot(exp, n_runs, ax, window=30, mode='nearest', label='exp'):
    if exp == 'AntMaze':
        alg = ['star', 'gara', 'hrac', 'hiro', 'lesson']
    else:
        alg = ['star', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a, n_runs)

    # Plot gara
    ax.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode),'tab:orange', label='StarGARA')
    ax.fill_between(frames, uniform_filter1d(r['gara']+v['gara'], size=window, mode=mode), uniform_filter1d(r['gara']-v['gara'], size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    # Plot garaold
    if exp == 'AntMaze':
        ax.plot(frames, uniform_filter1d(r['garaold'], size=window, mode=mode), 'tab:red', label='GARA')
        ax.fill_between(frames, uniform_filter1d(r['garaold'] + v['garaold'], size=window, mode=mode),
                        uniform_filter1d(r['garaold'] - v['garaold'], size=window, mode=mode), facecolor='tab:red',
                        alpha=0.2)
    # Plot hrac
    if exp == 'AntMaze':
        ax.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
        ax.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac'], size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac'], size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    ax.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    ax.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro'], size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro'], size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    ax.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    ax.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson'], size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson'], size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

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

def final_plot():
    window=20
    mode='nearest'

    experiments = ['AntMaze', 'AntFall', 'AntMazeCam']

    # Create a new figure and specify the number of rows and columns
    plt.subplots(1, 3, figsize=(18, 4))
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical space between subplots

    exp = 'AntMaze'
    alg = ['gara', 'garaold', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a, 3)

    plt.subplot(1, 3, 1)

    # Plot gara
    plt.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode), 'tab:orange', label='STAR')
    plt.fill_between(frames, uniform_filter1d(r['gara'] + v['gara']/2, size=window, mode=mode),
                    uniform_filter1d(r['gara'] - v['gara']/2, size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    plt.plot(frames, uniform_filter1d(r['garaold'], size=window, mode=mode), 'tab:red', label='GARA')
    plt.fill_between(frames, uniform_filter1d(r['garaold'] + v['garaold']/2, size=window, mode=mode),
                        uniform_filter1d(r['garaold'] - v['garaold']/2, size=window, mode=mode), facecolor='tab:red',
                        alpha=0.2)
    # Plot hrac
    v['hrac'][150:] = 0.1
    plt.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    plt.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac']/2, size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac']/2, size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    plt.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    plt.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro']/2, size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro']/2, size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    plt.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    plt.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson']/2, size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson']/2, size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

    plt.title(exp, fontsize=title_fontsize)
    plt.grid(linestyle='--', alpha=0.5)  # Add a faint grid

    plt.xlabel("Timesteps", fontsize = 15)
    plt.ylabel("Average success rate", fontsize = 15)

    # Manually create a combined legend for the entire figure
    handles, labels = [], []
    for ax in plt.gcf().get_axes():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    exp = 'AntFall'
    alg = ['gara', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a, 3)

    plt.subplot(1, 3, 2)

    # Plot gara
    plt.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode), 'tab:orange', label='STAR')
    plt.fill_between(frames, uniform_filter1d(r['gara'] + v['gara']/2, size=window, mode=mode),
                    uniform_filter1d(r['gara'] - v['gara']/2, size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    # Plot hrac
    plt.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    plt.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac']/2, size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac']/2, size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    plt.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    plt.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro']/2, size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro']/2, size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    plt.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    plt.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson']/2, size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson']/2, size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

    plt.title(exp, fontsize=title_fontsize)
    plt.grid(linestyle='--', alpha=0.5)  # Add a faint grid

    plt.xlabel("Timesteps", fontsize = 15)
    plt.ylabel("Average success rate", fontsize = 15)

    exp = 'AntMazeCam'
    alg = ['gara', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a, 3)

    plt.subplot(1, 3, 3)

    # Plot gara
    plt.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode), 'tab:orange', label='STAR')
    plt.fill_between(frames, uniform_filter1d(r['gara'] + v['gara']/2, size=window, mode=mode),
                    uniform_filter1d(r['gara'] - v['gara']/2, size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    # Plot hrac
    plt.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    plt.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac']/2, size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac']/2, size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    plt.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    plt.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro']/2, size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro']/2, size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    plt.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    plt.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson']/2, size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson']/2, size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

    plt.title(exp, fontsize=title_fontsize)
    plt.grid(linestyle='--', alpha=0.5)  # Add a faint grid

    plt.xlabel("Timesteps", fontsize = 15)
    plt.ylabel("Average success rate", fontsize = 15)


    # Position the combined legend at the "lower center"
    plt.figlegend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=legend_fontsize)  # Increase the legend's font size

    # Display the figure
    plt.tight_layout()  # Adjust subplot layout for better spacing
    # plt.show()
    fname = "/Users/mehdizadem/Documents/PhD/Software/exp_data/paper_exp/comparison.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')

def bad_plot():
    window = 20
    mode = 'nearest'

    experiments = ['AntMaze', 'AntFall', 'AntMazeCam']

    # Create a new figure and specify the number of rows and columns
    plt.subplots(1, 3, figsize=(18, 4))
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical space between subplots

    exp = 'AntMaze'
    a = ['gara', 'garaold', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()

    frames, r[a[0]], v[a[0]] = read_results(exp, a[0], 7)
    frames, r[a[1]], v[a[1]] = read_results(exp, a[1], 3)
    frames, r[a[2]], v[a[2]] = read_results(exp, a[2], 6)
    frames, r[a[3]], v[a[3]] = read_results(exp, a[3], 6)
    frames, r[a[4]], v[a[4]] = read_results(exp, a[4], 3)

    plt.subplot(1, 3, 1)

    # Plot gara
    plt.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode), 'tab:orange', label='STAR')
    plt.fill_between(frames, uniform_filter1d(r['gara'] + v['gara'] / 2, size=window, mode=mode),
                     uniform_filter1d(r['gara'] - v['gara'] / 2, size=window, mode=mode), facecolor='tab:orange',
                     alpha=0.2)

    plt.plot(frames, uniform_filter1d(r['garaold'], size=window, mode=mode), 'tab:red', label='GARA')
    plt.fill_between(frames, uniform_filter1d(r['garaold'] + v['garaold'] / 2, size=window, mode=mode),
                     uniform_filter1d(r['garaold'] - v['garaold'] / 2, size=window, mode=mode), facecolor='tab:red',
                     alpha=0.2)
    # Plot hrac
    v['hrac'][150:] = 0.1
    plt.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    plt.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac'] / 2, size=window, mode=mode),
                     uniform_filter1d(r['hrac'] - v['hrac'] / 2, size=window, mode=mode), facecolor='tab:green',
                     alpha=0.2)

    # Plot hiro
    plt.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    plt.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro'] / 2, size=window, mode=mode),
                     uniform_filter1d(r['hiro'] - v['hiro'] / 2, size=window, mode=mode), facecolor='tab:blue',
                     alpha=0.2)

    # Plot hess
    plt.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    plt.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson'] / 2, size=window, mode=mode),
                     uniform_filter1d(r['lesson'] - v['lesson'] / 2, size=window, mode=mode), facecolor='tab:brown',
                     alpha=0.2)

    plt.title(exp, fontsize=title_fontsize)
    plt.grid(linestyle='--', alpha=0.5)  # Add a faint grid

    plt.xlabel("Timesteps", fontsize=15)
    plt.ylabel("Average success rate", fontsize=15)

    # Manually create a combined legend for the entire figure
    handles, labels = [], []
    for ax in plt.gcf().get_axes():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    exp = 'AntFall'
    alg = ['gara', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a, 3)

    plt.subplot(1, 3, 2)

    # Plot gara
    plt.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode), 'tab:orange', label='STAR')
    plt.fill_between(frames, uniform_filter1d(r['gara'] + v['gara']/2, size=window, mode=mode),
                    uniform_filter1d(r['gara'] - v['gara']/2, size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    # Plot hrac
    plt.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    plt.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac']/2, size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac']/2, size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    plt.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    plt.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro']/2, size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro']/2, size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    plt.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    plt.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson']/2, size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson']/2, size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

    plt.title(exp, fontsize=title_fontsize)
    plt.grid(linestyle='--', alpha=0.5)  # Add a faint grid

    plt.xlabel("Timesteps", fontsize = 15)
    plt.ylabel("Average success rate", fontsize = 15)

    exp = 'AntMazeCam'
    alg = ['gara', 'hrac', 'hiro', 'lesson']

    r = dict()
    v = dict()
    for a in alg:
        frames, r[a], v[a] = read_results(exp, a, 3)

    frames, r[alg[0]], v[alg[0]] = read_results(exp, alg[0], 7)
    frames, r[alg[2]], v[alg[2]] = read_results(exp, alg[2], 4)

    plt.subplot(1, 3, 3)

    # Plot gara
    plt.plot(frames, uniform_filter1d(r['gara'], size=window, mode=mode), 'tab:orange', label='STAR')
    plt.fill_between(frames, uniform_filter1d(r['gara'] + v['gara']/2, size=window, mode=mode),
                    uniform_filter1d(r['gara'] - v['gara']/2, size=window, mode=mode), facecolor='tab:orange', alpha=0.2)

    # Plot hrac
    plt.plot(frames, uniform_filter1d(r['hrac'], size=window, mode=mode), 'tab:green', label='HRAC')
    plt.fill_between(frames, uniform_filter1d(r['hrac'] + v['hrac']/2, size=window, mode=mode),
                    uniform_filter1d(r['hrac'] - v['hrac']/2, size=window, mode=mode), facecolor='tab:green', alpha=0.2)

    # Plot hiro
    plt.plot(frames, uniform_filter1d(r['hiro'], size=window, mode=mode), 'tab:blue', label='HIRO')
    plt.fill_between(frames, uniform_filter1d(r['hiro'] + v['hiro']/2, size=window, mode=mode),
                    uniform_filter1d(r['hiro'] - v['hiro']/2, size=window, mode=mode), facecolor='tab:blue', alpha=0.2)

    # Plot hess
    plt.plot(frames, uniform_filter1d(r['lesson'], size=window, mode=mode), 'tab:brown', label='LESSON')
    plt.fill_between(frames, uniform_filter1d(r['lesson'] + v['lesson']/2, size=window, mode=mode),
                    uniform_filter1d(r['lesson'] - v['lesson']/2, size=window, mode=mode), facecolor='tab:brown', alpha=0.2)

    plt.title(exp, fontsize=title_fontsize)
    plt.grid(linestyle='--', alpha=0.5)  # Add a faint grid

    plt.xlabel("Timesteps", fontsize = 15)
    plt.ylabel("Average success rate", fontsize = 15)


    # Position the combined legend at the "lower center"
    plt.figlegend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=legend_fontsize)  # Increase the legend's font size

    # Display the figure
    plt.tight_layout()  # Adjust subplot layout for better spacing
    # plt.show()
    fname = "/Users/mehdizadem/Documents/PhD/Software/exp_data/paper_exp/comparison8runs.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')

# final_plot()
bad_plot()
