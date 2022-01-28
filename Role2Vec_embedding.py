from karateclub import Role2Vec
import networkx as nx
import numpy as np
import time
from Node2Vec_embedding import return_embeddings

start_time = time.time()

#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'walk_number': 10, 'walk_length': 15, 'dimensions': 128, 'window_size': 5, 
              'epochs': 1, 'workers': 7}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)

embeddings = return_embeddings(Role2Vec, parameters)
np.save('data/embedding_diff2vec_wn{:d}_wl{:d}_d{:d}_ws{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions'], parameters['window_size']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))

