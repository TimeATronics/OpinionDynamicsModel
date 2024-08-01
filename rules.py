import random
import csv
import itertools
from opdyn.Categories import fuzzy_cat
from opdyn.Population import Population
random.seed(1234)

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

    def roundToRange(self, value):
        # Workaround to ensure values do not overflow
        if value <= 0:
            return 0.0
        elif value >= 1:
            return 1.0
        else:
            return value

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

def main() -> None:
    header = ['Opinion(t)', 'Dissenter', 'Leader', 'NSI_Coeff', 'Avg_Opinion', 'Opinion(t+1)']
    with open('samples2.csv', 'w', encoding='UTF8',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    learning_rate = 0.5
    dis_percent = 0.25
    online_percent = 0.25
    leader_percent = 0.25
    leader_weight = 0.1
    conf_range = (0.1, 0.3)
    tol_range = (0.1, 0.3)
    n = Model(1, learn=learning_rate, dis_percent=dis_percent,
                leader_weight=leader_weight, conf_l=conf_range[0], conf_h=conf_range[1],
                tol_l=tol_range[0], tol_h=tol_range[1],
                onlinePercent=online_percent, leaderPercent=leader_percent, grid_size=3, distrib="Uniform")

    paramComb = [[0.0, 0.25, 0.5, 0.75, 1.0], [True, False], [True, False], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]] 
    combinations = list(itertools.product(*paramComb))
    for combination in combinations:
        param1, param2, param3, param4, param5 = combination[0], combination[1], combination[2], combination[3], combination[4]
        # Iterations:
        pos1, cell1 = list(n.popl.grid.items())[4]
        neighbors = cell1.getNeighbors()
        # Parameters:
        # 2. Dissenter - True / False
        cell1.setDissenter(param2)
        # 3. Leader - True / False
        cell1.setLeader(param3)
        # 4. NSI Coefficient - [0.0, 0.25, 0.5, 0.75, 1.0]:
        nsiValue = param4
        # 5. Average opinions - [0.0, 0.25, 0.5, 0.75, 1.0]:
        avgValue = param5
        for i in range(len(list(n.popl.grid.items()))):
            pos2, cell2 = list(n.popl.grid.items())[i]
            cell2.setOpinion(avgValue)
        # 1. Current Opinion
        cell1.setOpinion(param1)
        n.simulate(nsiValue)
        print("Op(t):", param1, " | Diss:", param2, " | Lead:", param3, " | NSI:", param4, " | Avg:", param5, " | Op(t+1):", cell1.getOpinion())
        with open('samples2.csv', 'a', encoding='UTF8',newline="") as f:
            writer=csv.writer(f)
            row = [cell1.getOpCat(param1), param2, param3, cell1.getNSICat(), cell1.getOpCat(param5), cell1.getOpCat(cell1.getOpinion())]
            writer.writerow(row)

if __name__ == "__main__":
    main()