from karateclub import Node2Vec
import networkx as nx
import numpy as np
import time

start_time = time.time()

def return_embeddings(G, model, parameters):
    """Creates embeddings for a given model

    Args:
        G: nx Graph.
        model: Graph embedding model from karateclub library.
        parameters (dict): Dictionary containing model parameters.

    Returns:
        np.ndarray: embeddings.
    """
    
    emb = model(**parameters)
    emb.fit(G)
    
    return emb.get_embedding()

# Create a graph
G = nx.read_edgelist('data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'walk_number': 10, 'walk_length': 15, 'dimensions': 64, 'workers': 7, 'window_size': 5}
#--------------------------------------------------------------------------------------#

n2v_embeddings = return_embeddings(G, Node2Vec, parameters)
np.save('data/embedding_n2v_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions'], parameters['window_size']), 
        n2v_embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
