import random
import numpy as np

from opdyn.Categories import fuzzy_cat
from opdyn.Model import Model
import opdyn.Helpers as Helpers

random.seed(1234)

def main() -> None:
    learning_rate = 0.5
    dis_percent = 0.25
    online_percent = 0.25
    leader_percent = 0.25
    leader_weight = 0.1
    conf_range = (0.1, 0.3)
    tol_range = (0.1, 0.3)
    n = Model(10000, learn=learning_rate, dis_percent=dis_percent,
                leader_weight=leader_weight, conf_l=conf_range[0], conf_h=conf_range[1],
                tol_l=tol_range[0], tol_h=tol_range[1],
                onlinePercent=online_percent, leaderPercent=leader_percent, grid_size=20, distrib="Beta")
    vis1 = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis1)):
        for col in range(len(vis1[0])):
            vis1[row][col] = round(fuzzy_cat["opinion"][n.popl.grid.get((row, col)).getOpCat(n.popl.grid.get((row, col)).getOpinion())], 2)

    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = round(n.popl.grid.get((row, col)).getOpinion(), 2)
    print(np.matrix(vis1))
    Helpers.plotHeatMap(vis1, "Initial Configuration")

    initial_op_ls = []
    for i in vis1:
        initial_op_ls.extend(i)
    Helpers.plot_finalOpinions_dist(initial_op_ls)

    n.simulate()
    print("\n\n")
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = n.popl.grid.get((row, col)).getOpinion()
    final_opinions_ls = []
    for row in range(len(vis)):
        final_opinions_ls.extend(vis[row])
    n.plot_opinions_over_time(final_opinions_ls)
    size_population = n.popl.grid_size * n.popl.grid_size

    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = round(n.popl.grid.get((row, col)).getOpinion(), 2)
    vis1 = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis1)):
        for col in range(len(vis1[0])):
            vis1[row][col] = round(fuzzy_cat["opinion"][n.popl.grid.get((row, col)).getOpCat(n.popl.grid.get((row, col)).getOpinion())], 2)
    print(np.matrix(vis1))
    Helpers.plotHeatMap(vis1, "Final Configuration")
    flattened_matrix = np.matrix(vis).flatten()

    Helpers.plot_finalOpinions_dist(final_opinions_ls)

    hhi = Helpers.metrics(final_opinions_ls, size_population)
    print("#####")
    print("HHI: ", hhi)
    print("Standard Deviation: ", round(np.std(flattened_matrix), 4))
    print("Mean: ", round(np.mean(flattened_matrix), 4))
    print("Median: ", np.median(vis))
    print("#####")

if __name__ == "__main__":
    main()