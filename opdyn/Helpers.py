import collections
import matplotlib.pyplot as plt
import numpy as np

def plotHeatMap(vis, title) -> None:
    # Plots a 2D heat map displaying all cells and their opinions
    plt.clf()
    plt.imshow(vis, origin='lower')
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title(title)
    plt.colorbar()
    plt.show()

def get_clusters(final_opinions_ls):
    # HHI helper function
    lastevolutions_opinions = final_opinions_ls
    count_lastevolutions_opinions = collections.Counter(lastevolutions_opinions)
    return count_lastevolutions_opinions

def metrics(final_opinions_ls, population_size):
    # HHI helper function
    cluster_dict = get_clusters(final_opinions_ls)
    cluster_sizes = list(cluster_dict.values())
    cluster_sizes = [i for i in cluster_sizes if i > 1]
    hhi = sum([(i/population_size)**2 for i in cluster_sizes])
    return np.round(hhi, 3)

def print_metrics(final_opinions_ls, population_size):
    # Prints HHI
    hhi = metrics(final_opinions_ls, population_size)
    print("HHI" + "     ===>  ", hhi)