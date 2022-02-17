from karateclub import Node2Vec
import networkx as nx
import numpy as np
import time
from tqdm import tqdm

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
parameters = {'walk_number': 10, 'walk_length': 15, 'dimensions': 64, 'window_size': 5, 'workers': 3}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)
#--------------------------------------------------------------------------------------#
G = nx.read_weighted_edgelist('data/unique_citation_graph_train.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
print(list(G.edges(data=True))[0])
#--------------------------------------------------------------------------------------#

embeddings = return_embeddings(G, Node2Vec, parameters)
np.save('data/embedding_n2v_unique_citation_graph_train_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions'], parameters['window_size']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
