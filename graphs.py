import random
import numpy as np
import matplotlib.pyplot as plt

from opdyn.Model import Model
import opdyn.Helpers as Helpers

random.seed(1234)

def main() -> None:
    val1 = [round(i, 3) for i in list(np.linspace(0, 1, 50))]
    val2 = [0.0, 0.25, 0.5, 0.75, 1.0]
    data = {}
    for dis in val2:
        subdata = []
        for val in val1:
            learning_rate = val
            dis_percent = 0.25
            online_percent = 0.25
            leader_percent = dis
            leader_weight = 0.1
            conf_range = (0.1, 0.3)
            tol_range = (0.1, 0.3)
            n = Model(1000, learn=learning_rate, dis_percent=dis_percent,
                    leader_weight=leader_weight, conf_l=conf_range[0], conf_h=conf_range[1],
                    tol_l=tol_range[0], tol_h=tol_range[1],
                    onlinePercent=online_percent, leaderPercent=leader_percent, grid_size=20, distrib="Uniform")
            vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
            for row in range(len(vis)):
                for col in range(len(vis[0])):
                    vis[row][col] = round(n.popl.grid.get((row, col)).getOpinion(), 2)
            n.simulate()
            for row in range(len(vis)):
                for col in range(len(vis[0])):
                    vis[row][col] = n.popl.grid.get((row, col)).getOpinion()
            final_opinions_ls = []
            for row in range(len(vis)):
                final_opinions_ls.extend(vis[row])

            size_population = n.popl.grid_size * n.popl.grid_size
            Helpers.print_metrics(final_opinions_ls, size_population)

            vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
            for row in range(len(vis)):
                for col in range(len(vis[0])):
                    vis[row][col] = round(n.popl.grid.get((row, col)).getOpinion(), 2)
            flattened_matrix = np.matrix(vis).flatten()
            hhi = Helpers.metrics(final_opinions_ls, size_population)
            print("#####")
            print("LEARNING RATE = ", val, " DIS_% = ", dis)
            print("Standard Deviation: ", round(np.std(flattened_matrix), 4))
            print("Mean: ", round(np.mean(flattened_matrix), 4))
            print("Median: ", np.median(vis))
            print("#####")
            subdata.append(hhi)
        data[dis] = subdata
    print(data)
    x = val1
    print(x)
    label = "lead = {num}"
    for i in range(5):
        plt.plot(x, data[val2[i]], label=label.format(num = val2[i]))
    plt.xlabel('Learning Rate')
    plt.ylabel('HHI')
    plt.title('HHI vs Learning Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()