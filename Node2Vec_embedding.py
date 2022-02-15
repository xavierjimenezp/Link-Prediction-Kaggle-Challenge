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
parameters = {'walk_number': 10, 'walk_length': 15, 'dimensions': 64, 'window_size': 5, 'workers': 5}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)
#--------------------------------------------------------------------------------------#
G = nx.read_weighted_edgelist('data/unique_co_authors_graph.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
print(list(G.edges(data=True))[0])
#--------------------------------------------------------------------------------------#
numeric_indices = [index for index in range(G.number_of_nodes())]
node_indices = sorted([node for node in G.nodes()])
if not numeric_indices == node_indices:
    max_nodes = max(G.nodes())
    missing_list = list(set([index for index in range(max_nodes)]) - set(node_indices))
    print("The node indexing is wrong. But the graph was complete by node not connected\n")
    for i in tqdm(missing_list, desc='Waiting for completion'):
        G.add_edge(i,i,weight = 0)
if max_nodes != 137861:
    print("Error")
#--------------------------------------------------------------------------------------#

embeddings = return_embeddings(G, Node2Vec, parameters)
np.save('data/embedding_n2v_unique_co_authors_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions'], parameters['window_size']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
