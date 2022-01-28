
from karateclub import GraphWave
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
parameters = {'sample_number': 200, 'step_size': 0.1, 'heat_coefficient': 1.0, 'approximation': 100}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)

embeddings = return_embeddings(GraphWave, parameters)
np.save('data/embedding_graphwave_sn{:d}_ss{:f}_hc{:f}_a{:d}'.format(parameters['sample_number'], parameters['step_size'],
                                                               parameters['heat_coefficient'], parameters['approximation']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
