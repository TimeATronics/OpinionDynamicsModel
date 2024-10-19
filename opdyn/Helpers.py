import collections
import matplotlib.pyplot as plt
import numpy as np

from opdyn.Categories import fuzzy_cat

def plotHeatMap(vis, title) -> None:
    # Plots a 2D heat map displaying all cells and their opinions
    plt.clf()
    plt.imshow(vis, origin='lower')
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_finalOpinions_dist(final_opinions_ls):
    # Displays a bar chart of final Opinions over fuzzy categories 

    opinion_cat = fuzzy_cat["opinion"]
    dist = {key: 0 for key in opinion_cat.keys()}
    categories = list(opinion_cat.keys())
    values = list(opinion_cat.values())
    num_cat = len(values)

    for op in final_opinions_ls:
        fuzzy_cat_idx = min(range(num_cat), key=lambda i: abs(values[i] - op))
        dist[categories[fuzzy_cat_idx]] += 1

    x_labels = list(dist.keys())   
    y_values = list(dist.values()) 

    plt.bar(x_labels, y_values, width=0.7) 
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart of Opinion distribution')
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