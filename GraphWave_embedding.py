
from karateclub import GraphWave
import networkx as nx
import numpy as np
import time
from Node2Vec_embedding import return_embeddings

start_time = time.time()

# Create a graph
G = nx.read_edgelist('data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)


#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'sample_number': 200, 'step_size': 0.1, 'heat_coefficient': 1.0, 'approximation': 100}
#--------------------------------------------------------------------------------------#


embeddings = return_embeddings(G, GraphWave, parameters)
np.save('data/embedding_graphwave_sn{:d}_ss{:f}_hc{:f}_a{:d}'.format(parameters['sample_number'], parameters['step_size'],
                                                               parameters['heat_coefficient'], parameters['approximation']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
