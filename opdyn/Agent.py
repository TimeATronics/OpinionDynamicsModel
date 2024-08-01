import random
import numpy as np
from opdyn.Categories import fuzzy_cat

class Agent:
    """
    Agent Class. An Agent is a cell in the 2D lattice (population)
    with the following parameters:
    01. opinion: a value between 0 and 1
    02. pos: a 2-tuple containing the x and y coordinates of the agent in the grid
    03. delta: a value between 0 and 1 (distinctiveness factor)
    04. grid_size: number of cells on each side of the grid
    05. k: a value between 0 and 1 (adjustment rate, for calculation of delta)
    06. tolerance: a value between 0 and 1, determines willingness to interact
    07. dissenter: True / False (determines whether cell can negatively interact)
    08. is_leader: True / False (a leader has a higher accessibility)
    09. confidence_threshold: a value between 0 and 1 (willingness to change opinion upon interaction)
    10. nsi: NSI coefficient (a fuzzified consolidated parameter) (0 - 1)
    11. distantNeighbors: cells at nth radius considered to be online neighbors
    12. connectivity: radius at which cell can interact with online neighbors at (1, 2, 3, ...)
    13. accessibility: max. number of cells at nth radius which can be interacted with (0 - 1)
    14. onlineAccess: True / False (whether an agent is connected / disconnected)
    15. radius: radius value for connectivity (1, 2, 3, ...)
    16. fuzzy_opinion: fuzzy degrees of membership for cell opinion
    17. fuzzy_avg_opinion: fuzzy degrees of membership for avg. opinion of confidence set cells
    18. fuzzy_nsi: fuzzy degrees of membership for nsi coefficient
    """
 
    def __init__(self, opinion, pos, delta, grid_size, nsi,
                 k, tolerance=0.0, dissenter=False, conf=0.5,
                 is_leader=False, media_influence=False, distantNeighbors=[],
                 radius=2, onlineAccess=False, accessibility=0.5) -> None:
        self.opinion = opinion
        self.posx = pos[0]
        self.posy = pos[1]
        self.delta = delta
        self.grid_size = grid_size
        self.k = k
        self.tolerance = tolerance
        self.dissenter = dissenter
        self.is_leader = is_leader
        self.confidence_threshold = conf
        self.nsi = nsi
        self.distantNeighbors = distantNeighbors
        self.connectivity = radius
        self.accessibility = accessibility
        self.onlineAccess = onlineAccess
        self.radius = radius
        self.fuzzy_opinion = {"Strongly Disagree": 0.0, "Disagree": 0.0,
                              "Neutral": 0.0, "Agree": 0.0, "Strongly Agree": 0.0}
        self.fuzzy_avg_opinion = {"Strongly Disagree": 0.0, "Disagree": 0.0,
                              "Neutral": 0.0, "Agree": 0.0, "Strongly Agree": 0.0}
        self.fuzzy_nsi = {"Non-conforming" : 0.0, "Slightly non-conforming": 0.0,
                          "Neutral": 0.0, "Slightly Conforming": 0.0, "Conforming": 0.0}

    def getNeighbors(self) -> list:
        # Returns Moore neighbors (excluding the cell itself)
        row, col = self.posx, self.posy
        neighbors = []
        for drow in [-1, 0, 1]:
            for dcol in [-1, 0, 1]:
                if drow == 0 and dcol == 0:
                    continue
                new_row = (row + drow) % self.grid_size
                new_col = (col + dcol) % self.grid_size
                neighbors.append((new_row, new_col))
        return neighbors

    def setDistantNeighbors(self):
        # Gets neighbors at only the nth radius from the cell and
        # adds them to the distant neighbor list according to accessibility
        neighbors = []
        radius = self.connectivity
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0 or not (0 <= (self.posx + dx) % self.grid_size < self.grid_size
                                                and 0 <= (self.posy + dy) % self.grid_size):
                    continue
                if abs(dx) <= radius and abs(dy) <= radius and (abs(dx) + abs(dy) == radius):
                    neighbors.append(((self.posx + dx) % self.grid_size, (self.posy + dy) % self.grid_size))
        self.distantNeighbors = neighbors[:int(self.accessibility * len(neighbors))]

    def gaussian(self, x, mean, sigma):
        # Gaussian fuzzifier (membership function)
        return round(np.exp(-0.5 * ((x - mean) / sigma) ** 2), 2)

    def fuzzify(self, param, paramValue, sigma=0.1):
        # Assigns degrees of membership to a parameter being fuzzified
        memberships = {}
        for category, mean in fuzzy_cat[param].items():
            memberships[category] = self.gaussian(paramValue, mean, sigma)
        return memberships

    def fuzzify_opinion(self, sigma=0.1):
        # Fuzzify opinions
        self.fuzzy_opinion = self.fuzzify("opinion", self.getOpinion(), sigma)

    def fuzzify_avg_opinion(self, value, sigma=0.1):
        # Fuzzifies avg. opinion of cells in confidence set
        self.fuzzy_avg_opinion = self.fuzzify("avg_opinion", value, sigma)
    
    def fuzzify_nsi(self, value, sigma=0.1):
        # Fuzzifies the NSI coefficient
        self.fuzzy_nsi = self.fuzzify("nsi_coeff", value, sigma)

    def defuzzify_opinion(self):
        # Fuzzy opinion -> Crisp opinion
        numerator = sum(mean * self.fuzzy_opinion[category] for category, mean in fuzzy_cat["opinion"].items())
        denominator = sum(self.fuzzy_opinion.values())
        self.setOpinion((numerator / denominator if denominator != 0 else self.getOpinion()))

    def defuzzify_nsi(self):
        # Fuzzy NSI coeff. -> Crisp NSI coeff.
        numerator = sum(mean * self.fuzzy_nsi[category] for category, mean in fuzzy_cat["nsi_coeff"].items())
        denominator = sum(self.fuzzy_nsi.values())
        self.setNSI((numerator / denominator if denominator != 0 else self.nsi))

    def getOpinion(self) -> float:
        return self.opinion
    def setOpinion(self, val) -> None:
        self.opinion = round(val, 2)
    def getPosition(self) -> list:
        return [self.posx, self.posy]
    def getDelta(self) -> int:
        return self.delta
    def setDelta(self, delta) -> None:
        self.delta = delta
    def setLeader(self, val) -> None:
        # If leadership is removed, change accessibility to a lower value
        if val == True:
            self.is_leader = True
            self.accessibility = random.random()**2
            self.accessibility /= (self.accessibility + 1)
        else:
            self.is_leader = False
            self.accessibility = random.random() ** (1 / 2)
    def remLeader(self) -> None:
        self.is_leader = False
        self.accessibility = random.random() ** (1 / 2)
    def setNSI(self, nsi) -> None:
        self.nsi = nsi
    def getNSICat(self):
        # Get closest fuzzy category for the crisp NSI value
        closest = min([0.00, 0.25, 0.50, 0.75, 1.00], key=lambda x: abs(x - self.nsi))
        category = list(fuzzy_cat["nsi_coeff"].keys()) [list(fuzzy_cat["nsi_coeff"].values()).index(closest)]
        return category
    def getOpCat(self, value):
        # Get closest fuzzy category for the crist opinion value
        closest = min([0.00, 0.25, 0.50, 0.75, 1.00], key=lambda x: abs(x - value))
        category = list(fuzzy_cat["opinion"].keys()) [list(fuzzy_cat["opinion"].values()).index(closest)]
        return category
    def setDissenter(self, val) -> None:
        self.dissenter = val
    def remDissenter(self) -> None:
        self.dissenter = False
    def checkDissenter(self) -> bool:
        return self.dissenter
    def setTolerance(self, tol) -> None:
        self.tolerance = tol
    def getTolerance(self) -> float:
        return self.tolerance