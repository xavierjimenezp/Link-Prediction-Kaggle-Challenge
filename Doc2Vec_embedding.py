import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

start_time = time.time()

a_file = open("data/abstract_preprocessed.pkl", "rb")
abstracts = pickle.load(a_file)
a_file.close()

#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'vector_size':128, 'window':5,
                'min_count':2, 'epochs':100, 'workers':7}
#--------------------------------------------------------------------------------------#


doc = []
for i in range(len(abstracts)):
    doc.append(abstracts[i])

# learns the embeddings of each abstract
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(doc)]
del doc
d2v = Doc2Vec(tagged_data, **parameters)

d2v.save("data/abstracts_embedding_doc2vec_vs{:d}_w{:d}_mc{:d}_e{:d}".format(parameters['vector_size'], parameters['window'],
                                                                               parameters['min_count'], parameters['epochs']))

print("Embedding learned")
print("--- %s seconds ---" % (time.time() - start_time))
