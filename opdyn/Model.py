import random
import matplotlib.pyplot as plt
import numpy as np
from opdyn.Categories import fuzzy_cat
from opdyn.Population import Population

class Model:
    """
    Model Class. This model is based on the HK model, Normative Social Influence,
    online communication through distant neighbors and other novelties through
    fuzzy cellular automata.   
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
        for agent in range(self.popl.grid_size * self.popl.grid_size): plt.plot(time, self.opinion_of_agents_over_time[agent])
        plt.xlabel("Time Steps")
        plt.ylabel("Opinion")
        plt.legend()
        plt.title("Opinion vs. Time")
        plt.show()

    def roundToRange(self, value):
        # Workaround to ensure values do not overflow
        if value <= 0:
            return 0.0
        elif value >= 1:
            return 1.0
        else:
            return value

    def update(self) -> None:
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
                nsi = cell1.k * round(self.popl.getIdealOpinion(cell1) - cell1.fuzzy_opinion[category], 2)
                cell1.setNSI(nsi)
                cell1.fuzzify_nsi(nsi)
                for cat2 in fuzzy_cat["nsi_coeff"]:
                    cell1.fuzzy_opinion[category] = self.roundToRange((cell1.fuzzy_opinion[category] * (1 - self.popl.leader_weight) + cell1.fuzzy_opinion[category] * self.popl.leader_weight) + cell1.fuzzy_nsi[cat2])

        # Case 2: If the current cell is a dissenter:
        if cell1.checkDissenter():
            for category in fuzzy_cat["opinion"]:
                # NSI coefficient
                nsi = cell1.k * round(self.popl.getIdealOpinion(cell1) - cell1.fuzzy_opinion[category], 2)
                cell1.setNSI(nsi)
                cell1.fuzzify_nsi(nsi)
                for cat2 in fuzzy_cat["nsi_coeff"]:
                    cell1.fuzzy_opinion[category] = self.roundToRange(cell1.fuzzy_opinion[category] - cell1.fuzzy_nsi[cat2])

        # Case 3: If the current cell is not a dissenter:
        else:
            for category in fuzzy_cat["opinion"]:
                nsi = cell1.k * round(self.popl.getIdealOpinion(cell1) - cell1.fuzzy_opinion[category], 2)
                cell1.setNSI(nsi)
                cell1.fuzzify_nsi(nsi)
                for cat2 in fuzzy_cat["nsi_coeff"]:
                    cell1.fuzzy_opinion[category] = self.roundToRange(cell1.fuzzy_opinion[category] + cell1.fuzzy_nsi[cat2])

        # Defuzzification to use opinion and nsi coeff. values for next iteration:
        cell1.defuzzify_opinion()
        cell1.defuzzify_nsi()
        # Update distinctiveness factor of the current cell:
        cell1.setDelta(self.popl.getNextDelta(cell1))

    def simulate(self) -> None:
        for t in range(self.timeSteps):
            self.update()
            self.get_agent_opinions()
            self.grid_opinion_over_time[t] = self.grid_op.copy()