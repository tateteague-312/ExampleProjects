import itertools
import random
import numpy as np
import networkx as nx
from MonteCarlo import MonteCarlo as MC
import matplotlib.pyplot as plt

class RandomNetwork(MC):
    def __init__(self,graph, randomNerworkGenerator): 
        '''Takes in a base graph as well as the desired random network generator; 'base', 'erdos renyi', or'barbasi albert' '''
        self.rawgraph = graph
        self.rn = randomNerworkGenerator
        self.numNodes = len(graph.nodes)
        self.numEdges = len(graph.edges)


    def erdosRenyi(self):
        '''Takes combinations of different nodes to create new edges and then randomly samples from list to get same number as before, modified from https://github.com/CambridgeUniversityPress/FirstCourseNetworkScience/blob/master/tutorials/Chapter%205%20Tutorial.ipynb'''
        G = nx.Graph()
        possible_edges = itertools.combinations(self.rawgraph.nodes, 2)
        edges_to_add = random.sample(list(possible_edges), self.numEdges)
        G.add_edges_from(edges_to_add)  
        
        self.erdosRenyiGraph = G


    def barbasiAlbert(self):
        ''' Builds Barabasi Albert Random graph based off base graph node count'''
        self.barbasiAlbertGraph = nx.barabasi_albert_graph(n=self.numNodes, m=1, seed=np.random.randint(1, 100))

    def stochastBlockModel(self, s,p):
        self.SBM_Random_Network = nx.stochastic_block_model(s, p, seed=np.random.randint(1, 100))
        

    def createPartitionMap(self, partition):
        '''PARAMS: partition RETURNS: partiion map; Creates a partition map to break the network into clusters'''
        partition_map = {}
        for idx, cluster_nodes in enumerate(partition):
            for node in cluster_nodes:
                partition_map[node] = idx
        return partition_map

    def girvan_newman_partition(self, graph, numSteps):
        '''PARAMS: graph object and # of clusters RETURNS clustered partitions that can be fed in as node colors to nx.draw(); Method of generating partitions and returns clustered partitions of network'''
        partition = list(nx.community.girvan_newman(graph))[numSteps - 2]
        partitionMap = self.createPartitionMap(partition)

        return [partitionMap[i] for i in graph.nodes()], partition

    def drawGraph(self, graph,title,node_colors = None, options = {'node_size':50,'linewidths':2}):
        
        plt.figure(figsize=(15,15))
        nx.draw_networkx(graph,with_labels=False,node_color = node_colors, **options)
        plt.title(title, fontsize=40)
        plt.show()

    ##################################################################
    ################## Simulate Once #################################
    def SimulateOnce(self):
        '''Generates summary statistics after N runs of different graph generations'''
        if self.rn == 'erdos renyi':
            self.erdosRenyi()
            G = self.erdosRenyiGraph
        elif self.rn == 'barbasi albert':
            self.barbasiAlbert()
            G = self.barbasiAlbertGraph
        elif self.rn == 'base':
            G = self.rawgraph
        else:
            print("Unknown Network type, defaulting to base graph. Please pass 'Erdos Renyi' or 'Barbasi' upon class initiation")
            G = self.rawgraph

        ### Degree Centrality
        degrees = [G.degree(j) for j in G.nodes]
        avgDegree = np.mean(degrees)

        ### Betweenness Centrality
        betweenness = list(nx.centrality.betweenness_centrality(G).values())
        avgBetweenness = np.mean(betweenness)
        
        ### Avg Shortest Path
        avgConnectedness = []
        for i in (G.subgraph(i).copy() for i in nx.connected_components(G)):
            avgConnectedness.append(nx.average_shortest_path_length(i))
        avgConnectedness = np.mean(avgConnectedness)

        return avgConnectedness, avgBetweenness, avgDegree


        

##################################################################################
#### Full run can be commented out during analysis phase with Jupyter Notebook####
##################################################################################

## Raw Network
print('Raw Network')
amzGraph = nx.read_edgelist('C:/Users/tate5/Documents/Git/Monte Carlo Final Project - Network Graphs/amz.txt')
print(nx.info(amzGraph))

rn = RandomNetwork(amzGraph, 'base')

con,deg, bet  = rn.RunSimulation(simCount=10)
print('For 10 simulations with the std Amz graph we have the following metrics:')
print(f'The mean connectedness(average of the avg shortest path):\nleft: {round(con[0],2)} mean:{round(con[1],2)} right:{round(con[2],2)}')
print(f'\nThe mean degree(# of edges connected to a node):\nleft: {round(deg[0],2)} mean:{round(deg[1],2)} right:{round(deg[2],2)}')
print(f'\nThe mean betweenness(number of times a node lies on the shortest path between other nodes):\nleft: {round(bet[0],4)} mean:{round(bet[1],4)} right:{round(bet[2],4)}')

rn.drawGraph(rn.rawgraph, 'Raw Network Graph')


## Erdos Renyi
print('Erdos Renyi')

rn_renyi = RandomNetwork(amzGraph, 'erdos renyi')
con,deg, bet  = rn_renyi.RunSimulation(simCount=10)
print('For 10 simulations with the std Amz graph we have the following metrics:')
print(f'The mean connectedness(average of the avg shortest path):\nleft: {round(con[0],2)} mean:{round(con[1],2)} right:{round(con[2],2)}')
print(f'\nThe mean degree(# of edges connected to a node):\nleft: {round(deg[0],2)} mean:{round(deg[1],2)} right:{round(deg[2],2)}')
print(f'\nThe mean betweenness(number of times a node lies on the shortest path between other nodes):\nleft: {round(bet[0],4)} mean:{round(bet[1],4)} right:{round(bet[2],4)}')
rn_renyi.drawGraph(rn_renyi.erdosRenyiGraph, 'Erdos Renyi Graph')


## Barbasi Albert
print('Barbasi Albert')

rn_barbasi = RandomNetwork(amzGraph, 'barbasi albert')
con,deg, bet  = rn_barbasi.RunSimulation(simCount=10)
print('For 10 simulations with the std Amz graph we have the following metrics:')
print(f'The mean connectedness(average of the avg shortest path):\nleft: {round(con[0],2)} mean:{round(con[1],2)} right:{round(con[2],2)}')
print(f'\nThe mean degree(# of edges connected to a node):\nleft: {round(deg[0],2)} mean:{round(deg[1],2)} right:{round(deg[2],2)}')
print(f'\nThe mean betweenness(number of times a node lies on the shortest path between other nodes):\nleft: {round(bet[0],4)} mean:{round(bet[1],4)} right:{round(bet[2],4)}')
rn_barbasi.drawGraph(rn_barbasi.barbasiAlbertGraph, 'Barbasi Albert Graph')


## Stochastic Black Model
print('Stochastic Black Model on Raw Network')
s = [round(len(rn_barbasi.rawgraph) * .30), round(len(rn_barbasi.rawgraph) * .30), round(len(rn_barbasi.rawgraph) * .40)]
p = [[0.25, 0.05, 0.02],
     [0.05, 0.35, 0.07],
     [0.02, 0.07, 0.40]]

rn_barbasi.stochastBlockModel(s,p)
rn_barbasi.drawGraph(rn_barbasi.SBM_Random_Network, 'Stochastic Block Model')

## Best Partition Clustering
print('Best Partition on Raw Network')

nodeColor,partition = rn.girvan_newman_partition(rn.rawgraph, 4)
rn.drawGraph(rn.rawgraph,'Best Partition Groupings(raw network, partitions = 12)',node_colors=nodeColor)

nodeColor,partition = rn.girvan_newman_partition(rn.rawgraph, 4)
rn.drawGraph(rn.rawgraph,'Best Partition Groupings(raw network, partitions = 12)',node_colors=nodeColor)

        