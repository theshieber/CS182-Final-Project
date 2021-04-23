from graphics import *
import random
import math
import itertools
import sets
import copy
import numpy as np
import matplotlib.pyplot as plt
import time


NUMNODES = 10
SPREAD = 750
VEHICLE_CAPACITY = 3
MAXCOST = 999999999999

# generates a graph in the form of a list that we will turn into our adjacency list
def generate_graph_list():
	initialNode = (random.randint(0, SPREAD), random.randint(0, SPREAD))
	createdNodes = [initialNode]
	graphList = []

	for i in range(NUMNODES - 1):

		prevNodeInd = random.randint(0, len(createdNodes)-1)
		prevNode = createdNodes[prevNodeInd]
		newNode = (random.randint(0, SPREAD), random.randint(0, SPREAD))

		while newNode in createdNodes:
			newNode = (random.randint(0, SPREAD), random.randint(0, SPREAD))

		graphList.append((prevNode, newNode))
		createdNodes.append(newNode)

		nodeList = createdNodes

	return (graphList, nodeList)

# makes an adjacency list from the graph list
def make_adjacency(graphList):
	graphAdjacency = {}

	for i in range(len(graphList)):
		nodeA = graphList[i][0]
		nodeB = graphList[i][1]

		if nodeA in graphAdjacency:
			graphAdjacency[nodeA].append(nodeB)
			graphAdjacency[nodeB] = [nodeA]
		elif nodeB in graphAdjacency:
			graphAdjacency[nodeB].append(nodeA)
			graphAdjacency[nodeA] = [nodeB]
		else:
			graphAdjacency[nodeA] = [nodeB]
			graphAdjacency[nodeB] = [nodeA]

	return graphAdjacency

# draws the graph using the python graphics library
def draw_graph(graphAdjacency, nodeStart, nodes, clusters):
	win = GraphWin('RoadGraph', SPREAD, SPREAD)

	(pickups, deliveries) = genPickupDelivery(nodes, 2)

	for node, nodeList in graphAdjacency.iteritems():
		for i in range(len(nodeList)):

			(ptAx, ptAy) = node
			(ptBx, ptBy) = nodeList[i]

			ptA = Point(ptAx, ptAy)
			ptB = Point(ptBx, ptBy)

			ptAcirc = Circle(ptA, 3)
			ptAcirc.setFill('black')
			ptBcirc = Circle(ptB, 3)
			ptBcirc.setFill('black')

			edge = Line(ptA, ptB)

			label = Text(ptA, str(node))
			label.setTextColor('red')

			if node == nodeStart:
				ptAcirc = Circle(ptA, 5)
				ptAcirc.setFill('blue')
			if nodeList[i] == nodeStart:
				ptBcirc = Circle(ptB, 5)
				ptBcirc.setFill('blue')

			ptAcirc.draw(win)
			ptBcirc.draw(win)
			edge.draw(win)
			label.draw(win)

	win.mainloop()

# builds the dictionary that houses all costs
def make_cost_dict(graphAdjacency):
	graphCosts = {}

	for node, nodeList in graphAdjacency.iteritems():
		for i in range(len(nodeList)):

			cost = round(get_euclid_dist(node, nodeList[i]), 2)

			graphCosts[(node, nodeList[i])] = cost

	return graphCosts

# prints the adjacency graph in an aesthetically pleasing way
def pretty_print(graphAdjacency, graphCosts):
	print "<<<< ADJACENCY LIST >>>>\n"
	for node, nodeList in graphAdjacency.iteritems():

		rightSide = ""

		for i in range(len(nodeList)):
			rightSide += str(nodeList[i]) + " : " + str(graphCosts[(node, nodeList[i])]) + ", "

		print str(node) + " ---> " + str(rightSide) + "\n"
	print "<<<< ADJACENCY LIST >>>>\n"

# gets the euclidean distrance between 2 points
def get_euclid_dist(pointA, pointB):
	dist = math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)
	return dist

# gets the cost of a path
def get_cost_of_path(graphAdjacency, graphCostsComplete, path, graphCosts):
	cost = 0
	for i in range(len(path) - 1):
		nodeA = path[i]
		nodeB = path[i + 1]
		if nodeB in graphAdjacency[nodeA]:
			cost += graphCosts[(nodeA, nodeB)]
		else:
			cost += graphCostsComplete[(nodeA, nodeB)]
	return cost

# runs bfs to get the shortest path between points. was used for testing purposes.
def bfs_shortest_path(graphAdjacency, nodeStart, nodeGoal):
	explored = []
	queue = [[nodeStart]]

	if nodeStart == nodeGoal:
		return queue
		
	while queue:
		path = queue.pop(0)
		node = path[-1]

		if node not in explored:
			neighbours = graphAdjacency[node]

			for neighbour in neighbours:
				new_path = list(path)
				new_path.append(neighbour)
				queue.append(new_path)

				if neighbour == nodeGoal:
					return new_path

			explored.append(node)

# runs dijkstras algorithm on the graph. returns a predecessors array and also a distances array
def dijkstra(graph, nodeStart, graphCosts, visited, distances, predecessors):

	if visited == sets.Set(['']): 
		distances[nodeStart]=0	
	for neighbor in range(len(graph[nodeStart])):
		if neighbor not in visited:
			new_distance = distances[nodeStart] + graphCosts[(nodeStart, (graph[nodeStart])[neighbor])]
			if new_distance < distances.get(graph[nodeStart][neighbor], 100000):
				distances[graph[nodeStart][neighbor]] = new_distance
				predecessors[graph[nodeStart][neighbor]] = nodeStart
	visited.add(nodeStart)
	unvisited={}
	for k in graph:
		if not k in visited:
			unvisited[k] = distances.get(k,100000)

	if unvisited == {}:
		return (predecessors, distances)
	x=min(unvisited, key=unvisited.get)
	
	return dijkstra(graph, x, graphCosts, visited, distances, predecessors)

# converts an incomplete graph to a complete one where if 2 nodes do not have a direct edge in the
# original graph, instead the shortest path length is that edge cost in the new graph.
def toCompleteGraph(graphAdjacency, nodeList, graphCosts):
	newGraphAdjacency = copy.deepcopy(graphAdjacency)
	newGraphCosts = copy.deepcopy(graphCosts)
	currentDistance = []
	predecessors = {}

	for firstNode in nodeList:
		runDijk = dijkstra(graphAdjacency, firstNode, graphCosts, sets.Set(['']), {}, {})
		currentDistance = runDijk[1]
		currentPredeccessors = runDijk[0]
		predecessors[firstNode] = currentPredeccessors
		for nextNode in currentDistance:
			newGraphCosts.update({(firstNode, nextNode):currentDistance[nextNode]})
			newGraphCosts.update({(nextNode, firstNode):currentDistance[nextNode]})
			newGraphAdjacency[firstNode].append(nextNode)
			newGraphAdjacency[nextNode].append(firstNode)

	return (newGraphAdjacency, newGraphCosts, predecessors)

# expands a path from the complete graph to the original graph
def expand_path(predecessors, path):
	predecessorsFull = []
	for i in range(len(path) - 1):
		firstNode = path[i]
		nextNode = path[i + 1]
		firstNodePred = predecessors[firstNode]
		predecessorsFull.append(firstNode)
		nextPred = firstNodePred[nextNode]
		predecessorsTemp = []
		while not nextPred == firstNode:
			predecessorsTemp.append(nextPred)
			nextPred = firstNodePred[nextPred]
		predecessorsFull = predecessorsFull + list(reversed(predecessorsTemp))
	predecessorsFull.append(path[-1])
	return predecessorsFull

# generates a random path
def randomPath(nodeList, startNode):
	randList = nodeList[1:]
	random.shuffle(randList)
	randPath = [startNode] + randList + [startNode]
	return randPath

'''
TSP solvers
'''

# runs the brute force algorithm on the graph
def tsp_brute_force(graphAdjacency, graphCompleteCosts, graphCosts, nodeList, startNode):
	permuList = list(itertools.permutations(nodeList[1:]))
	bestPath = None
	bestPathCost = 99999999999
	count = 0.
	for pathSub in permuList:
		path = list(pathSub)
		path = [startNode] + path
		path.append(startNode)
		count += 1.
		newPathCost = 0
		for i in range(NUMNODES-1):
			newPathCost = get_cost_of_path(graphAdjacency, graphCompleteCosts, path, graphCosts)
		if newPathCost < bestPathCost:
			bestPathCost = newPathCost
			bestPath = path

	return (bestPath, bestPathCost)

# gets a successor for simulated annealing
def get_path_successor(path, successorSize):
	newPath = copy.deepcopy(path)
	i = random.randint(1, successorSize-1)
	j = random.randint(1, successorSize-1)
	while j == i:
		j = random.randint(1, successorSize-1)
	temp = newPath[i]
	newPath[i] = newPath[j]
	newPath[j] = temp
	return newPath

# plots the results for simulated annealing graph using the python pyplot module
def plot_sim_anneal_delta(deltaList, deltaInds):
	plt.plot(deltaInds, deltaList)

# runs simulated annealing with random initial paths. (random restart)
def simulated_annealing_random_restart(nodeList, startNode, graphAdjacency, graphAdjacencyCompleteCosts, graphAdjacencyCosts, iterations, restarts):
	bestCost = 1000000
	bestPath = []
	for i in range(restarts):
		initSimAnnealPath = randomPath(nodeList, startNode)
		(curSimAnnealPath, curSimAnnealPathCost) = simulated_annealing(initSimAnnealPath, graphAdjacency, graphAdjacencyCompleteCosts, graphAdjacencyCosts, iterations)
		if curSimAnnealPathCost < bestCost:
			bestCost = curSimAnnealPathCost
			bestPath = curSimAnnealPath

	return (bestPath, bestCost)

# runs simulated annealing
def simulated_annealing(path, graphAdjacency, graphAdjacencyCompleteCosts, graphAdjacencyCosts, iterations):
	successorSize = len(path)
	current = get_path_successor(path, successorSize)
	currentCost = get_cost_of_path(graphAdjacency, graphAdjacencyCompleteCosts, current, graphAdjacencyCosts)
	Inds = []
	deltaList = []
	successorCostList = []
	def T(i):
		return 10**10 * 0.8**np.floor((i + 5000)/300)

	for t in range(0, iterations):
		Temp = T(t)
		successor = get_path_successor(current, successorSize)
		successorCost = get_cost_of_path(graphAdjacency, graphAdjacencyCompleteCosts, successor, graphAdjacencyCosts)

		deltaE = currentCost - successorCost
		rand = random.random()

		if deltaE > 0:
			Inds.append(t)
			deltaList.append(deltaE)
			successorCostList.append(successorCost)

			current = successor
			currentCost = successorCost
		
		elif rand < math.exp(deltaE/Temp):
			current = successor
			currentCost = successorCost
		
	plot_sim_anneal_delta(successorCostList, Inds)

	return (current, currentCost)

# runs the greedy algorithm on the graph
def greedyTSP(graphCompleteCosts, nodeList, startNode):
	currNode = startNode
	unvisited = copy.deepcopy(nodeList)
	unvisited.remove(currNode)
	def buildGreedyPath(graphCompleteCosts, currNode, unvis, path, pathCost):
		if unvis == []:
			return (path, pathCost)
		else:
			bestMove = None
			currMin = 9999999
			for node in unvis:
				if graphCompleteCosts[(currNode, node)] < currMin:
					bestMove = node
					currMin = graphCompleteCosts[(currNode, node)]
			path.append(bestMove)
			pathCost += currMin
			unvis.remove(bestMove)
			return buildGreedyPath(graphCompleteCosts, bestMove, unvis, path, pathCost)
	(greedyPath, greedyPathCost) =  buildGreedyPath(graphCompleteCosts, currNode, unvisited, [currNode], 0)
	# to return to start node
	greedyPathCost += graphCompleteCosts[(startNode, greedyPath[-1])]
	greedyPath.append(startNode)
	return (greedyPath, greedyPathCost)

"""
Clustering
"""
# gets andgle between 2 points
def thetaAB(nodeA, nodeB):
	opposite = nodeB[1] - nodeA[1]
	adjacent = nodeB[0] - nodeA[0]
	return math.atan2(opposite, adjacent)

# gets all of the thetas
def allThetas(startNode, node0, nodeList):
	theta0 = thetaAB(startNode, node0)
	thetaList = []
	for i in range(len(nodeList)):
		theta = theta0 - thetaAB(startNode, nodeList[i])
		thetaList.append(theta)
	return thetaList

# gets the index of the next angle
def nextThetaIndex(visited, thetaList, nodeList):
	currMin = 99999999
	currNext = None
	for i in range(len(thetaList)):
		if currMin > thetaList[i]:
			if not nodeList[i] in visited:
				currMin = thetaList[i]
				currNext = i
	return currNext 

# implements the swep algorithm
def sweep(nodeList, startNode, graphAdjacency, graphCosts, numVehicles):
	node0 = startNode
	while node0 == startNode:
		node0i = random.randint(0, NUMNODES-1)
		node0 = nodeList[node0i]
	clusters = []
	thetaList = allThetas(startNode, node0, nodeList)
	visited = [startNode,node0]
	(completeGraphAdjacency, completeGraphCosts, predecessors) = toCompleteGraph(graphAdjacency, nodeList, graphCosts)	 
	while len(visited) != NUMNODES:
		if numVehicles == 0:
			raise ValueError('Not Enough Vehicles')
		capacity = VEHICLE_CAPACITY
		currCost = 0
		cluster = []
		if len(visited) == 2:
			cluster.append(node0)
			capacity += -1
			currCost += 2 * completeGraphCosts[(startNode, node0)]
		while capacity != 0 and currCost < MAXCOST and len(visited) != NUMNODES:
			nextIndex = nextThetaIndex(visited, thetaList, nodeList)
			cluster.append(nodeList[nextIndex])
			capacity += -1
			currCost += 2 * completeGraphCosts[(startNode, nodeList[nextIndex])]
			visited.append(nodeList[nextIndex])
		clusters.append(cluster)
		numVehicles += -1
	return clusters

#prints a diction in an aesthetically pleasing way
def pretty_dict_print(dictionary):
	 for k, v in dictionary.iteritems():
	 	print str(k) + " --> " +  str(v) + '"\n'

# run everything
def main():

	''' -- either create a random graph of NUMNODES or use the example graph -- '''
	new_random_graph = True
	new_example_graph = False

	''' -- variables for running algorithms -- '''
	nodeList = []
	exampleList = []
	graphList = []
	graphAdj = {}
	graphAdjCosts = {}
	graphAdjComplete = {}
	graphAdjCompleteCosts = {}
	predecessors = []
	startNode = None
	pathPerTruckBrute = {}
	pathPerTruckGreedy = {}
	pathPerTruckSimAnneal = {}

	if new_random_graph:
		(graphList, nodeList) = generate_graph_list()
		graphAdj = make_adjacency(graphList)
		graphAdjCosts = make_cost_dict(graphAdj)
		(graphAdjComplete, graphAdjCompleteCosts, predecessors) = toCompleteGraph(graphAdj, nodeList, graphAdjCosts)
		startNode = nodeList[0]
	elif new_example_graph:
		nodeList = [(100, 400), (50, 250), (100, 100), (400, 300), (400, 150), (200, 200)]
		graphAdj = {(50, 250):[(100, 400)],
					(100, 400):[(50,250), (400, 300), (200, 200)],
					(400, 300):[(100, 100), (100, 400), (400, 150)],
					(100, 100):[(400, 300), (200, 200)],
					(400, 150):[(400, 300)],
					(200, 200):[(100, 100), (100, 400)]}
		exampleList = [(100, 400), (400, 150), (50, 250)]
		graphAdjCosts = make_cost_dict(graphAdj)
		(graphAdjComplete, graphAdjCompleteCosts, predecessors) = toCompleteGraph(graphAdj, nodeList, graphAdjCosts)
		startNode = nodeList[0]

	''' -- print whatever graph is being used -- '''
	pretty_print(graphAdj, graphAdjCosts)

	print "<<<< TSP SOLUTIONS >>>>\n"

	''' -- run and print brute force -- '''
	(bruteForcePath, bruteForcePathCost) = tsp_brute_force(graphAdj, graphAdjCompleteCosts, graphAdjCosts, nodeList, startNode)
	bruteForcePathExpanded = expand_path(predecessors, bruteForcePath)

	print "  << brute Force >>"
	print "    bruteForcePath: " + str(bruteForcePath)
	print "    bruteForchPathExpanded: " + str(bruteForcePath)
	print "    bruteForcePathCost: " + str(bruteForcePathCost)
	print "\n<<<< TSP SOLUTIONS >>>>\n"

	''' -- run and print greedy force -- '''
	(greedyPath, greedyPathCosts) = greedyTSP(graphAdjCompleteCosts, nodeList, startNode)
	greedyPathExpanded = expand_path(predecessors, greedyPath)

	print "\n  << greedy >> "
	print "    greedyPath: " + str(greedyPath)
	print "    greedyPathExpanded: " + str(greedyPathExpanded)
	print "    greedyPathCost: " + str(greedyPathCosts)
	print "\n<<<< TSP SOLUTIONS >>>>\n"


	''' -- run and print simulated annealing and variants -- '''
	initSimAnnealPath = randomPath(nodeList, startNode)
	(simAnnealPath, simAnnealPathCost) = simulated_annealing(initSimAnnealPath, graphAdj, graphAdjCompleteCosts, graphAdjCosts, 30000)
	(simAnnealPathRR, simAnnealPathCostRR) = simulated_annealing_random_restart(nodeList, startNode, graphAdj, graphAdjCompleteCosts, graphAdjCosts, 30000, 10)
	(simAnnealPathGS, simAnnealPathCostGS) = simulated_annealing(greedyPath, graphAdj, graphAdjCompleteCosts, graphAdjCosts, 30000)
	simAnnealPathExpanded = expand_path(predecessors, simAnnealPath)
	print simAnnealPathExpanded
	simAnnealPathExpandedRR = expand_path(predecessors, simAnnealPathRR)
	simAnnealPathExpandedGS = expand_path(predecessors, simAnnealPathGS)

	print "\n  << simulated annealing >> "
	print "    simAnnealPath: " + str(simAnnealPath)
	print "    simAnnealPathExpanded: " + str(simAnnealPathExpanded)
	print "    simAnnealPathCost: " + str(simAnnealPathCost)
	print "    simAnnealPathRR: " + str(simAnnealPathRR)
	print "    simAnnealPathExpandedRR: " + str(simAnnealPathExpandedRR)
	print "    simAnnealPathCostRR: " + str(simAnnealPathCostRR)	
	print "    simAnnealPathGS: " + str(simAnnealPathGS)
	print "    simAnnealPathExpandedGS: " + str(simAnnealPathExpandedGS)
	print "    simAnnealPathCostGS: " + str(simAnnealPathCostGS)
	print "<<<< TSP SOLUTIONS >>>>\n"

	''' -- run sweep and clustering -- '''
	print "\n<<<< SWEEP SOLUTIONS >>>>\n"
	clusters = sweep(nodeList, startNode, graphAdj, graphAdjCosts, 10)
	print "CLUSTERS: " + str(clusters)
	for i in range(len(clusters)):
		clusterNodes = [startNode] + clusters[i]
		print "\n"
		print "Cluster " + str(i) + ":"
		print [startNode] + clusters[i]

		(bruteForceClusterPath, bruteForceClusterPathCost) = tsp_brute_force(graphAdj, graphAdjCompleteCosts, graphAdjCosts, clusterNodes, startNode)
		bruteForceClusterPathExpanded = expand_path(predecessors, bruteForceClusterPath)
		print "Brute Force: "
		print (bruteForceClusterPathCost, bruteForceClusterPathExpanded)
		pathPerTruckBrute[i] = (bruteForceClusterPathCost, bruteForceClusterPathExpanded)

		(greedyClusterPath, greedyClusterPathCost) = greedyTSP(graphAdjCompleteCosts, clusterNodes, startNode)
		greedyClusterPathExpanded = expand_path(predecessors, greedyClusterPath)
		print "Greedy: "
		print (greedyClusterPathCost, greedyClusterPathExpanded)
		pathPerTruckGreedy[i] = (greedyClusterPathCost, greedyClusterPathExpanded)
		
		(simAnnealClusterPath, simAnnealClusterPathCost) = simulated_annealing(clusterNodes, graphAdj, graphAdjCompleteCosts, graphAdjCosts, 30000)
		simAnnealClusterPathExpanded = expand_path(predecessors, simAnnealClusterPath + [startNode])
		cost = get_cost_of_path(graphAdj, graphAdjCompleteCosts, simAnnealClusterPathExpanded, graphAdjCosts)
		print "simAnnealPath: "
		print (cost, simAnnealClusterPathExpanded)
		pathPerTruckSimAnneal[i] = (cost, simAnnealClusterPathExpanded)

	print "\n<<<< SWEEP SOLUTIONS >>>>\n"

	''' -- uncomment to draw graph and to show simulated annealing graph -- '''
	#show sim anneal graph:
	#plt.show()
	#draw graph:
	#draw_graph(graphAdj, startNode, nodeList, clusters)

main()