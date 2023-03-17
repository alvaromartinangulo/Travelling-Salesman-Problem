import math
import graph
import random
import matplotlib.pyplot as plt

def EuclideanGeneration(minNumOfNodes,maxNumOfNodes):
    filenames = []
    for i in range(0, (maxNumOfNodes - minNumOfNodes) + 1):
        filename = "euclidean" + str(i)
        file = open(filename, "w")
        writer = [str(random.randint(0, 50)), str(random.randint(0, 50))]
        file.write(writer[0] + " " + writer[1] + "\n")
        increment = 50
        filenames.append(filename)
        for z in range(minNumOfNodes + i) :
            writer[1] = int(writer[1]) + increment
            for x in range(minNumOfNodes + i):
                writer[0] = int(writer[0]) + increment
                file.write(str(writer[0]) + " " + str(writer[1]) + "\n")
        file.close()
    return filenames
def runHeuristic():
    fileList = EuclideanGeneration(10,15)
    nodes = []
    swapHeuristicValues = []
    twoOptHeuristicValues = []
    greedyValues = []
    geneticValues = []
    for i in fileList:
        g = graph.Graph(-1, i)
        nodes.append(g.n)
        random.shuffle(g.perm)
        g.swapHeuristic(100)
        swapHeuristicValues.append(g.tourValue())
        print("Swap", g.tourValue())
        print("Swap", g.perm)
        random.shuffle(g.perm)
        g.TwoOptHeuristic(10)
        twoOptHeuristicValues.append(g.tourValue())
        random.shuffle(g.perm)
        g.Greedy()
        print(g.perm)
        print("Greedy", g.tourValue())
        greedyValues.append(g.tourValue())
        random.shuffle(g.perm)
        g.Genetic(1000,20,0.1,0.1)
        geneticValues.append(g.tourValue())
    print("Swap heurisitic", swapHeuristicValues)
    print("Two opt",twoOptHeuristicValues)
    print("greedy", greedyValues)
    print("Genetic", geneticValues)
    plt.plot(nodes, swapHeuristicValues, label = 'Swap Heuristic values')
    plt.plot(nodes, twoOptHeuristicValues, label = 'Two Heuristic values')
    plt.plot(nodes, greedyValues, label = 'Greedy values')
    plt.plot(nodes, geneticValues, label = 'Genetic values')
    plt.plot()
    plt.xlabel("Number of nodes")
    plt.ylabel("Tour value")
    plt.legend()
    plt.show()

runHeuristic()
