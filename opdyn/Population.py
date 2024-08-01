import random
from opdyn.Agent import Agent

class Population:
    """
    Population Class. A population is a configuration consisting of a 2D grid of
    Agents along with their parameters and values at a particular time step.
    Population-level parameters include:
    01. Uniform, Beta, Random: boolean values determining which distribution is
                               used to initialize opinions
    02. grid_size: number of cells on each side of the grid
    03. learning_rate: a value between 0 and 1 used to update delta
    04. dis_percent: % of dissenters within the population
    05. leader_percent: % of leaders within the population
    06. online_percent: % of online connected agents within the population
    07. leader_weight: a value between 0 and 1 which determines influence of leader
    08. conf_l, conf_h: min. and max. range limits for confidence threshold
    09. tol_l, tol_h: min. and max. range limits for tolerance
    """

    grid = {}
    def __init__(self, grid_size=10, Uniform=True,
                 Beta=False, Random=False,
                 learn=0.25, dis_percent=0.01, leader_weight=0.1, conf_l=0.1, conf_h=0.3,
                 tol_l=0, tol_h=0.15, onlinePercent=0.5, leaderPercent=0.5) -> None:
        self.Uniform = Uniform
        self.Beta = Beta
        self.Random = Random
        self.grid_size = grid_size
        self.learning_rate = learn
        self.dis_percent = dis_percent
        self.leader_weight = leader_weight
        self.conf_l = conf_l
        self.conf_h = conf_h
        self.tol_l = tol_l
        self.tol_h = tol_h
        self.onlinePercent = onlinePercent
        self.leaderPercent = leaderPercent
        self.createPopulation()
        self.setDissenters()
        self.setOnlineAcc()
        self.setLeaders()

    def createPopulation(self) -> None:
        # Initialize opinions and parameters of all Agents in the grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.grid[(row, col)] = Agent(opinion=self.createOpinion(), pos=[row, col],
                                              delta=self.createRandom(),
                                              grid_size=self.grid_size,
                                              nsi=0,
                                              k=self.createRandom(),
                                              dissenter=False,
                                              tolerance=round(random.uniform(self.tol_l, self.tol_h), 2),
                                              conf=round(random.uniform(self.conf_l, self.conf_h), 2),
                                              is_leader=False,
                                              distantNeighbors=[], radius=random.randint(1, 5),
                                              onlineAccess=False, accessibility=0)
                self.grid[(row, col)].setLeader(False)

    def setDissenters(self) -> None:
        # Set dissenters within the population based on % of dissenters
        totalDissenters = int(self.dis_percent * self.grid_size * self.grid_size)
        for i in range(totalDissenters):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[(x, y)].setDissenter(True)
    
    def setOnlineAcc(self) -> None:
        # Set online connected cells within the population based on % of online connected cells
        totalOnline = int(self.onlinePercent * self.grid_size * self.grid_size)
        for i in range(totalOnline):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[(x, y)].onlineAccess = True
            # online connected agents have distant neighbors
            self.grid[(x, y)].setDistantNeighbors()

    def setLeaders(self) -> None:
        # Set leader within the population based on % of leaders
        totalLeaders = int(self.leaderPercent * self.grid_size * self.grid_size)
        for i in range(totalLeaders):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[(x, y)].setLeader(True)

    def getNextOpinion(self, cell) -> int:
        # Based on conformity, returns next opinion of a cell
        return round(cell.getOpinion() + cell.k *
                     (round(self.getIdealOpinion(cell), 2) - cell.getOpinion()), 2)

    def getMeanOpinion(self, cell) -> float:
        # Returns mean opinion of all neighbors (including online)
        data = []
        neighbors = cell.getNeighbors()
        if cell.onlineAccess:
            neighbors.extend(cell.distantNeighbors)
        for i in range(4):
            data.append(self.grid[neighbors[i]].getOpinion())
        if len(data) == 0: return float('nan')
        return round(sum(data) / len(data), 2)

    def getSDOpinion(self, cell) -> int:
        # Returns standard deviation of opinions of all neighbors (including online)
        data = []
        neighbors = cell.getNeighbors()
        if cell.onlineAccess:
            neighbors.extend(cell.distantNeighbors)
        for i in range(4):
            data.append(self.grid[neighbors[i]].getOpinion())
        if not data: return None
        mean = sum(data) / len(data)
        squared_deviations = [pow(x - mean, 2) for x in data]
        variance = sum(squared_deviations) / len(data)
        return round(pow(variance, 0.5), 2)

    def getIdealOpinion(self, cell) -> float:
        # Based on conformity, returns ideal opinion of a cell
        return round(self.getMeanOpinion(cell) + cell.getDelta() * self.getSDOpinion(cell), 2)

    def getAvgDelta(self, cell) -> int:
        # Computes average distinctiveness factor of all neighbors (including distant neighbors)
        data = []
        neighbors = cell.getNeighbors()
        if cell.onlineAccess:
            neighbors.extend(cell.distantNeighbors)
        for i in range(4):
            data.append(self.grid[neighbors[i]].getDelta())
        return sum(data) / len(data)

    def getNextDelta(self, cell) -> int:
        # Based on NSI, updates delta of a cell to the its next possible value (closer to mean)
        data = []
        neighbors = cell.getNeighbors()
        if cell.onlineAccess:
            neighbors.extend(cell.distantNeighbors)
        for i in range(4):
            data.append(self.grid[neighbors[i]].getDelta())
        newDelta = min(max(cell.getDelta() +
                            self.learning_rate *
                            (self.getAvgDelta(cell) - cell.getDelta()), -5),
                            5)
        return int(newDelta)

    def createOpinion(self) -> float:
        # Computes opinion based on distribution used
        if self.Beta == True:
            return self.createBetaOpinion()
        if self.Uniform == True:
            return self.createUniformOpinion()
        if self.Random == True:
            return self.createRandomOpinion()

    def createBetaOpinion(self, alpha=2, beta=2) -> float:
        u1, u2 = random.random(), random.random()
        t1 = pow(u1, 1/(alpha-1))
        t2 = pow(u2, 1/(beta-1))
        sample = (t1 + t2) / (1 + t1 + t2)
        return round(sample, 2)

    def createUniformOpinion(self, low=0, high=1) -> float:
        return round(random.uniform(low, high), 2)

    def createRandom(self) -> float:
        value = random.random()
        while value <= 0 or value >= 1:
            value = random.random()
        return round(value, 2)

    def createRandomOpinion(self, low=0, high=1) -> float:
        value = random.random() * (high + abs(low)) + low
        while low > value or value > high:
            value = random.random() * (high + abs(low)) + low
        return round(value, 2)