import random
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from opdyn.Categories import fuzzy_cat
from opdyn.Population import Population
import opdyn.Helpers as Helpers
#random.seed(1234)

class Model:
    """
    Model Class which allows for constant values of all parameters
    """
    # opinions_over_time : stores opinions per time step for line plot
    opinions_over_time = {}
    def __init__(self, timeSteps, learn, dis_percent, leader_weight, conf_l, conf_h,
                 tol_l, tol_h, onlinePercent, leaderPercent, grid_size, distrib) -> None:
        if distrib == "Beta":
            self.Beta = True
            self.Uniform = False
            self.Random = False
        if distrib == "Uniform":
            self.Uniform = True
            self.Beta = False
            self.Random = False
        if distrib == "Random":
            self.Beta = False
            self.Uniform = False
            self.Random = True
        # initializing the population
        self.popl = Population(grid_size, self.Uniform, self.Beta, self.Random, learn, dis_percent, leader_weight, conf_l, conf_h,
                               tol_l, tol_h, onlinePercent, leaderPercent)
        self.grid_op = None
        self.timeSteps = timeSteps
        self.grid_opinion_over_time = {t : None for t in range(int(self.timeSteps))}
        self.opinion_of_agents_over_time = None

    def roundToRange(self, value):
        # Workaround to ensure values do not overflow
        if value <= 0:
            return 0.0
        elif value >= 1:
            return 1.0
        else:
            return value
    def get_agent_opinions(self):
        # Returns opinions of all agents in the population within self.grid_op
        if self.grid_op is None:
            self.grid_op = np.zeros([self.popl.grid_size,self.popl.grid_size])
        for pos, _ in np.ndenumerate(self.grid_op):
            self.grid_op[pos] = self.popl.grid[pos].opinion

    def plot_opinions_over_time(self, final_opinions_ls):
        # Displays a line plot of Opinion vs. Time
        time = list(self.grid_opinion_over_time.keys())
        opinions = [ op.flatten() for op in list(self.grid_opinion_over_time.values())]
        self.opinion_of_agents_over_time = [[opinions[t][pos] for t in time] for pos in range(len(opinions[0]))]
        for agent in range(self.popl.grid_size * self.popl.grid_size):
            plt.plot(time, self.opinion_of_agents_over_time[agent])
        plt.xlabel("Time Steps")
        plt.ylabel("Opinion")
        plt.legend()
        plt.title("Opinion vs. Time")
        plt.show()

    def update(self, nsiValue) -> None:
        # Transition Function / Local Rule

        # Fully Asynchronous Update (one cell selected at random at once)
        pos1, cell1 = random.choice(list(self.popl.grid.items()))
        neighbors = cell1.getNeighbors()
        # If cell is connected online:
        if cell1.onlineAccess:
            neighbors.extend(cell1.distantNeighbors)

        # Confidence set of the agent (based on HK model + tolerance from extended BCM)
        confidence_set = [i for i in neighbors if abs(self.popl.grid[i].getOpinion() - cell1.getOpinion())
                           <= cell1.confidence_threshold - cell1.getTolerance() * 2]

        # Calculation of degrees of membership for the average fuzzy opinion of all agents
        # in the confidence set of the current cell:
        avg_fuzzy_opinion = {category: 0 for category in fuzzy_cat["opinion"]}
        if confidence_set:        
            for i in confidence_set:
                self.popl.grid[i].fuzzify_opinion()
                for category in fuzzy_cat["opinion"]:
                    avg_fuzzy_opinion[category] += self.popl.grid[i].fuzzy_opinion[category]
            num_neighbors = len(confidence_set)
            for category in fuzzy_cat["opinion"]:
                avg_fuzzy_opinion[category] /= num_neighbors
            cell1.fuzzy_opinion = avg_fuzzy_opinion

        # Case 1: If current agent is a leader:
        if cell1.is_leader:
            for category in fuzzy_cat["opinion"]:
                nsi = nsiValue
                cell1.setNSI(nsi)
                cell1.fuzzify_nsi(nsi)
                for cat2 in fuzzy_cat["nsi_coeff"]:
                    cell1.fuzzy_opinion[category] = self.roundToRange((cell1.fuzzy_opinion[category] * (1 - self.popl.leader_weight) + cell1.fuzzy_opinion[category] * self.popl.leader_weight) + cell1.fuzzy_nsi[cat2])

        # Case 2: If the current cell is a dissenter:
        if cell1.checkDissenter():
            for category in fuzzy_cat["opinion"]:
                # NSI coefficient
                nsi = nsiValue
                cell1.setNSI(nsi)
                cell1.fuzzify_nsi(nsi)
                for cat2 in fuzzy_cat["nsi_coeff"]:
                    cell1.fuzzy_opinion[category] = self.roundToRange(cell1.fuzzy_opinion[category] - cell1.fuzzy_nsi[cat2])

        # Case 3: If the current cell is not a dissenter:
        else:
            for category in fuzzy_cat["opinion"]:
                nsi = nsiValue
                cell1.setNSI(nsi)
                cell1.fuzzify_nsi(nsi)
                for cat2 in fuzzy_cat["nsi_coeff"]:
                    cell1.fuzzy_opinion[category] = self.roundToRange(cell1.fuzzy_opinion[category] + cell1.fuzzy_nsi[cat2])

        # Defuzzification to use opinion and nsi coeff. values for next iteration:
        cell1.defuzzify_opinion()
        cell1.defuzzify_nsi()
        # Update distinctiveness factor of the current cell:
        cell1.setDelta(self.popl.getNextDelta(cell1))

    def simulate(self, nsiValue) -> None:
        for t in range(self.timeSteps):
            self.update(nsiValue)
            self.get_agent_opinions()
            self.grid_opinion_over_time[t] = self.grid_op.copy()

def main() -> None:


    learning_rate = 0.3
    dis_percent = 0.0#(3/400)
    online_percent = 0.0
    leader_percent = 0.0
    leader_weight = 0.1
    conf_range = (0.1, 0.3)
    tol_range = (0.1, 0.3)
    n = Model(4500, learn=learning_rate, dis_percent=dis_percent,
                leader_weight=leader_weight, conf_l=conf_range[0], conf_h=conf_range[1],
                tol_l=tol_range[0], tol_h=tol_range[1],
                onlinePercent=online_percent, leaderPercent=leader_percent, grid_size=25, distrib="Uniform")

    paramComb = [[0.0, 0.25, 0.5, 0.75, 1.0], [True, False], [True, False], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]] 
    #combinations = list(itertools.product(*paramComb))
    #for combination in combinations:
#    param1, param2, param3, param4, param5 = combination[0], combination[1], combination[2], combination[3], combination[4]
        # Iterations:
    for (pos1, cell1) in list(n.popl.grid.items()):
        #neighbors = cell1.getNeighbors()
            # Parameters:
            # 2. Dissenter - True / False
        #cell1.setDissenter(False)
            # 3. Leader - True / False
        #cell1.setLeader(False)
            # 4. NSI Coefficient - [0.0, 0.25, 0.5, 0.75, 1.0]:
        #nsiValue = 0.5
            # 5. Average opinions - [0.0, 0.25, 0.5, 0.75, 1.0]:
        #avgValue = 
        cell1.setOpinion(1.0)
    #list(n.popl.grid.items())[0][1].setDissenter(True)
    #list(n.popl.grid.items())[0][1].setOpinion(0.75)
    #list(n.popl.grid.items())[1][1].setDissenter(True)
    #list(n.popl.grid.items())[1][1].setOpinion(0.25)
    #list(n.popl.grid.items())[20][1].setDissenter(True)
    #list(n.popl.grid.items())[20][1].setOpinion(1.0)
#    list(n.popl.grid.items())[21][1].setDissenter(True)
#    list(n.popl.grid.items())[2][1].setDissenter(True)
#    for i in range(len(list(n.popl.grid.items()))):
#        pos2, cell2 = list(n.popl.grid.items())[i]
#        cell2.setOpinion(avgValue)
        # 1. Current Opinion
#    cell1.setOpinion(param1)
    
    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = round(n.popl.grid.get((row, col)).getOpinion(), 2)
    Helpers.plotHeatMap(vis, "Init Configuration")
    n.simulate(0.5)
    #Helpers.plotHeatMap(vis, "Init Configuration")
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = n.popl.grid.get((row, col)).getOpinion()
    final_opinions_ls = []
    for row in range(len(vis)):
        final_opinions_ls.extend(vis[row])
    n.plot_opinions_over_time(final_opinions_ls)
    size_population = n.popl.grid_size * n.popl.grid_size
    Helpers.print_metrics(final_opinions_ls, size_population)

    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = round(n.popl.grid.get((row, col)).getOpinion(), 2)
    flattened_matrix = np.matrix(vis).flatten()
    Helpers.plotHeatMap(vis, "Final Configuration")
if __name__ == "__main__":
    main()


"""
Set all agents' opinion -> 1.0
for loop: % of dissenters increase from 0 to 100
(probably will converge to neutral)




Conclusion: by placing individual dissenters in a polarized population
through interaction over long time, everyone goes down to neutral
but dissenters (if not interacting) stick out of the population
and retain their original opinion

Dissenters do not change opinions in a majority population of non dissenters
In a population where everyone is a dissenter, the probability of 
change in their opinion is higher.

small populations converge quickly plot

Conclusion:
-> A population with no dissenters, leaders or online connected individuals coverges
   to a complete steady state (everyone's opinion = 0.5 / Neutral)
-> The rate of convergence to the steady state grows with increase in grid size
-> This is observed to be best described as a polynomial growth through a chi^2 
   goodness of fit test.

Population Size    Time Taken to converge to 0.5
1. 5                81, 124, 77, 44, 80, 93, 94, 139, 98, 84
2. 10               476, 527, 325, 415, 447, 457, 732, 534, 468, 481
3. 20               2410,2637,2098,3289,2086,3228,2899,2390,2567,2914
4. 30               5845,5115,6694,6224,5597,5124,6037,5431,5216,5991
5. 50               17334,19790,18886,20012,18056               
6. 3                9,24,22,35,40
7. 15               985,1122,1100,1285,1146
8. 25               3526,3990,3472,3583,3607

3 -> 26
5 -> 91.4
10 -> 486.2
15 -> 1127.6
20 -> 2651.8
25 -> 3635.6
30 -> 5727.4
50 -> 18815.6






"""


