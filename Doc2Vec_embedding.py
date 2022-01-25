import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

start_time = time.time()

# Read/create the preprocessed abstract of each paper
try:
    print('Loading abstracts')
    a_file = open("data/abstract_preprocessed.pkl", "rb")
    abstracts = pickle.load(a_file)
    a_file.close()
    print('Abstract already preprocessed')
except:
    # import nltk
    # nltk.download('stopwords')
    # nltk.download('punkt')
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from tqdm import tqdm 
    
    print('Preprocessing abstracts')
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    abstracts = dict()
    with open('data/abstracts.txt', 'r') as f:
        for line in tqdm(f):
            node, abstract = line.split('|--|')
            abstract = abstract.lower()
            # abstract = "".join([char for char in abstract if char not in string.punctuation])
            abstract = word_tokenize(abstract)
            abstract = [word for word in abstract if word not in stop_words]
            # abstract = [porter.stem(word) for word in abstract]
            abstracts[int(node)] = abstract
    a_file = open("data/abstract_preprocessed.pkl", "wb")
    pickle.dump(abstracts, a_file)
    a_file.close()
    print('Preprocessing Done')
    

#--------------------------------------------------------------------------------------#
#----------------------------# MODEL PARAMETERS TO CHANGE #----------------------------#
parameters = {'vector_size':128, 'window':5,
              'min_count':2, 'epochs':100, 'workers':7}
#--------------------------------------------------------------------------------------#

print("Start Embedding")

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
