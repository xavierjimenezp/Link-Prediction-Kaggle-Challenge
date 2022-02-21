from karateclub import Walklets
import networkx as nx
import numpy as np
import time
from pyrsistent import v
from tqdm import tqdm
import random

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

#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'walk_number': 10, 'walk_length': 80, 'dimensions': 64, 'window_size': 5, 'workers': 5}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)

<<<<<<< HEAD
G = nx.read_weighted_edgelist('data/unique_citation_graph_complete.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)

embeddings = return_embeddings(G, Walklets, parameters)
np.save('data/embedding_Walklets_unique_citation_graph_complete_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
=======
G = nx.read_weighted_edgelist('data/citation_graph_train.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)

embeddings = return_embeddings(G, Walklets, parameters)
np.save('data/embedding_Walklets_citation_graph_train_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
>>>>>>> 53ef7117bb4d508e14c3b04f783786ca889dae9b
                                                               parameters['dimensions'], parameters['window_size']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
