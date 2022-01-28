from karateclub import Node2Vec
import networkx as nx
import numpy as np
import time

start_time = time.time()

def return_embeddings(model, parameters):
    """Creates embeddings for a given model

    Args:
        G: nx Graph.
        model: Graph embedding model from karateclub library.
        parameters (dict): Dictionary containing model parameters.

    Returns:
        np.ndarray: embeddings.
    """
    
    emb = model(**parameters)
    G = nx.read_edgelist('data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
    emb.fit(G)
    
    return emb.get_embedding()


#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'walk_number': 10, 'walk_length': 80, 'dimensions': 64, 'window_size': 5, 'workers': 3}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)

n2v_embeddings = return_embeddings(Node2Vec, parameters)
np.save('data/embedding_n2v_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions'], parameters['window_size']), 
        n2v_embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
