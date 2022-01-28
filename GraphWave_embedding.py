
from karateclub import GraphWave
import networkx as nx
import numpy as np
import time
from Node2Vec_embedding import return_embeddings

start_time = time.time()

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
