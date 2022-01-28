from karateclub import DeepWalk
import networkx as nx
import numpy as np
import time
from Node2Vec_embedding import return_embeddings

start_time = time.time()



#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'walk_number': 20, 'walk_length': 80, 'dimensions': 128, 'workers': 7}
#--------------------------------------------------------------------------------------#
print('Learning embedding with parameters')
print(parameters)

dw_embeddings = return_embeddings(DeepWalk, parameters)
np.save('data/embedding_deepwalk_wn{:d}_wl{:d}_d{:d}'.format(parameters['walk_number'], parameters['walk_length'],
                                                               parameters['dimensions']), 
        dw_embeddings)

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))