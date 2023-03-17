import math
import random


def euclid(p,q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)

class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self,n,filename):
        file = open(filename, "r")
        # In the Euclidean TSP case.
        if n == -1:
            # temporary variable numOfNodes counts the nodes in the file, and
            # nodes temporarily stores the coordinates of the nodes to find
            # distances between them.
            numOfNodes = 0
            nodes = []
            for line in file:
                # Each node made of splitting the lines of the file, and
                # is formed of the x and y coordinates.
                node =  list(map(int, line.split()))
                nodes.append(node)
                numOfNodes += 1
            self.n = numOfNodes
            # Initialising the 2d array of distances, and setting the distances
            # with the euclid method provided.
            self.dists = [[0 for x in range(self.n)] for z in range(self.n)]
            for i in range (self.n):
                for z in range (self.n):
                    self.dists[i][z] = euclid(nodes[i], nodes[z])
        # In the general TSP case.
        else:
            self.n = n
            self.dists = [[0 for x in range(self.n)] for z in range(self.n)]
            for line in file:
                # each nodeDist is a list formed of the first node, the second
                # node and the distance between them.
                nodeDist = list(map(int, line.split()))
                self.dists[nodeDist[0]][nodeDist[1]] = nodeDist[2]
                self.dists[nodeDist[1]][nodeDist[0]] = nodeDist[2]
        file.close()
        # Setting the identity permutation of the length of the number of nodes.
        self.perm = [x for x in range(self.n)]


    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        cost = 0
        for node in self.perm:
            # If the node is not the last node, add the distance to the next
            # node.
            try:
                cost = cost + self.dists[self.perm[node]][self.perm[node + 1]]
            # If the node is the last node, add the distance from the last node
            # to the first node.
            except IndexError:
                cost = cost + self.dists[self.perm[node]][self.perm[0]]
        return cost

    # Helper function to switch indexes in the permutation tour array given any two indexes.
    # Realised later in the coursework that python had a built in way of doing
    # this by doing self.perm[i], self.perm[j] = self.perm[j], self.perm[i]
    # but decided to include this function as I thought this was still interesting.
    def swap(self,i,j):
        first = self.perm[i]
        self.perm[i] = self.perm[j]
        self.perm[j] = first

    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self,i):
        currentCost = self.tourValue()
        self.swap(i, ((i + 1) % self.n))
        if self.tourValue() >= currentCost:
            self.swap(i, ((i + 1) % self.n))
            return False
        else:
            return True

    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.
    def tryReverse(self,i,j):
        # Store the current permutation in a list to reverse the values in
        # self.perm
        currentPerm = list(self.perm)
        currentCost = self.tourValue()
        counter = 0
        while counter <= (j - i):
            self.perm[i + counter] = currentPerm[j - counter]
            counter += 1
        if self.tourValue() < currentCost:
            return True
        else:
            self.perm = currentPerm
            return False

    def swapHeuristic(self,k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            beroutetter = False
            count += 1
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self,k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(self.n-1):
                for i in range(j):
                    if self.tryReverse(i,j):
                        better = True


    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        currentTour = [self.perm[0]]
        currentIndex = 0
        # Loop until the tour built has the length of the number of nodes
        while len(currentTour) != self.n:
            # list comp for builiding a list of the nodes that have not
            # yet been used in the tour
            nodes = [node for node in self.perm if not node in currentTour]
            # first assume the best node is the one following the current node
            bestDist = self.dists[currentTour[currentIndex]][nodes[0]]
            bestNode = nodes[0]
            # compare all of the other nodes and set the best node and distances
            # accordingly.
            for node in nodes:
                currentDist = self.dists[currentTour[currentIndex]][node]
                if currentDist < bestDist:
                    bestNode = node
                    bestDist = currentDist
                if currentDist == bestDist:
                    if node < bestNode:
                        bestNode = node
                        bestDist = currentDist
            currentTour.append(bestNode)
            currentIndex += 1
        self.perm = currentTour

# Creating my own algorithm

    #Creates a population of random routes given a population size.
    def createInitialRoutes(self,initialPopSize):
        initialPopulation = []
        for i in range (initialPopSize):
            initialPopulation.append(random.sample(self.perm, len(self.perm)))
        return initialPopulation

    # Calculates the distances for the routes given a list of routes
    def calcDists(self, routes):
        dists = []
        for i in routes:
            self.perm = i
            dists.append(self.tourValue())
        return (dists)
    # Calculates the fitnesses for routes given the cost of the distances
    def calcFitness(self, costs):
        fitnesses = []
        for i in costs:
            # The lower the distance, the higher the fitness
            routeFitness = 1 / i
            fitnesses.append(routeFitness)
        return fitnesses

    # Creates a mating pool of routes given a population of routes, and the percentage
    # of the best routes that should be passed on to the next generation directly
    def createMatingPool(self, routes, bestRoutesPercentage):
        matingPool = []
        # For the best percentage of routes
        for i in range(int(bestRoutesPercentage * len(routes))):
            matingPool.append(routes[i])
        # For the rest of the routes add random routes to the mating pool
        for i in range(len(routes) - int(bestRoutesPercentage * len(routes))):
            matingPool.append(random.choice(routes))
        return matingPool

    # Given two parent routes, A child is formed by transferring part of one
    # route to the other one.
    def breed(self, parent1, parent2):
        child = []
        # Initialising the length of the route to transfer, and what index to start
        # transferring.
        geneLength = int(random.random() * len(parent1))
        geneStart = int(random.random() * (len(parent1) - geneLength))
        if geneStart == len(parent1):
            geneStart = len(paren1) - 1
        geneTransfer = parent1[geneStart : (geneStart + geneLength)]
        # The rest of the child will be the part of the other parent that is not
        # included in the gene to transfer
        restOfGene = [x for x in parent2 if not x in geneTransfer]
        for i in range(len(restOfGene)):
            if i == geneStart:
                child.extend(geneTransfer)
                child.append(restOfGene[i])
            else:
                child.append(restOfGene[i])
        return child

    # Creates the new generation of routes by passing the best routes directly
    # to the next generation, and using the breed function to create the rest
    # of the population
    def breedMatingPool(self, routes, bestRoutesPercentage):
        newRoutes = []
        for i in range(int(bestRoutesPercentage * len(routes))):
            newRoutes.append(routes[i])
        for i in range(int(len(routes) - int(len(routes) * bestRoutesPercentage))):
            randomParents = random.sample(routes, 2)
            child = self.breed(randomParents[0], randomParents[1])
            newRoutes.append(child)
        return(newRoutes)

    # Mutates members of the population based on a given probability.
    def mutatePopulation(self, routes, mutateProbability):
        mutatedRoutes = []
        for route in routes:
            if random.random() < mutateProbability:
                swap = random.sample(route, 2)
                # Swaps two random indexes of the route
                route[route.index(swap[0])], route[route.index(swap[1])] = route[route.index(swap[1])], route[route.index(swap[0])]
            mutatedRoutes.append(route)
        return mutatedRoutes

    def Genetic(self, numOfGenerations, initialPopSize, bestFromPopPercent, mutationRate):
        #Initialising the first population
        routes = self.createInitialRoutes(initialPopSize)
        costs = self.calcDists(routes)
        fitnesses = self.calcFitness(costs)
        population = sorted(list(zip(routes, costs, fitnesses)), key = lambda item: item[2], reverse=True)
        bestRoute = population[0]
        # Repeating the process for all of the generations
        for generation in range (numOfGenerations):
            routes = list(map(lambda x: x[0], population))
            matingPool = self.createMatingPool(routes, bestFromPopPercent)
            routes = self.breedMatingPool(matingPool, bestFromPopPercent)
            routes = self.mutatePopulation(routes, mutationRate)
            costs = self.calcDists(routes)
            fitnesses = self.calcFitness(costs)
            population = sorted(list(zip(routes, costs, fitnesses)), key = lambda item: item[2], reverse=True)
            if fitnesses[0] > bestRoute[2]:
                bestRoute = population[0]
        self.perm = bestRoute[0]
        return bestRoute[1]
