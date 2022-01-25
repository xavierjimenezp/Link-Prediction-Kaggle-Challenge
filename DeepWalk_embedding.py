from karateclub import DeepWalk
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
parameters = {'walk_number': 20, 'walk_length': 80, 'dimensions': 128, 'workers': 7}
#--------------------------------------------------------------------------------------#


dw_embeddings = return_embeddings(G, DeepWalk, parameters)
np.save('data/embedding_deepwalk_wn{:d}_wl{:d}_d{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions']), 
        dw_embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))