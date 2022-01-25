from karateclub import Diff2Vec
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
parameters = {'diffusion_number': 10, 'diffusion_cover': 10, 'dimensions': 128, 'window_size': 5, 
              'epochs': 1, 'workers': 7}
#--------------------------------------------------------------------------------------#


embeddings = return_embeddings(G, Diff2Vec, parameters)
np.save('data/embedding_diff2vec_dn{:d}_dc{:d}_d{:d}_ws{:d}'.format(parameters['diffusion_number'], parameters['diffusion_cover'],
                                                               parameters['dimensions'], parameters['window_size']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))