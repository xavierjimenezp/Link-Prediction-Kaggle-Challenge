from karateclub import Diff2Vec
import networkx as nx
import numpy as np
import time
from Node2Vec_embedding import return_embeddings

start_time = time.time()

#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'diffusion_number': 10, 'diffusion_cover': 10, 'dimensions': 128, 'window_size': 5, 
              'epochs': 1, 'workers': 7}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)

embeddings = return_embeddings(Diff2Vec, parameters)
np.save('data/embedding_diff2vec_dn{:d}_dc{:d}_d{:d}_ws{:d}'.format(parameters['diffusion_number'], parameters['diffusion_cover'],
                                                               parameters['dimensions'], parameters['window_size']), 
        embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))