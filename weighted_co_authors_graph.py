# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:01:10 2022

@author: sacha
"""

import utils
import copy
import numpy as np
from FastNode2Vec_embedding import fast_node2vec
from gensim.models import KeyedVectors
import re
from Node2Vec_embedding import node2vec
from scipy.sparse import lil_matrix,csr_matrix,save_npz, load_npz
import itertools

def create_files_co_authors_graph(method):
    """
    
    Parameters
    ----------
    method : string : "classic", "first_letter_first_name"
        Give the method we will use to to transform input names of authors
        to another representation of this name.
        
        classic : implies no modification of the names give in input
        first_letter_first_name : creta new names with the first letter 
        of the first name and the full family name
    Returns
    -------
    authors_node : dict
        key(str): author name, value(int): node
    method_authors : dict
        key(int) : papers number, value(list of string) : authors of paper
    creation : bool
        just to check is the creation is ok

    """
    creation:bool = True
    if method=='classic':
        try:
            method_authors = \
                    utils.load_authors(utils.FILE_PATH_AUTHORS,method)
            authors_node=utils.load_authors_node(utils.FILE_PATH_AUTHORS_NODE,method)
        except:
            method_authors = utils.load_authors(utils.FILE_PATH_AUTHORS,'classic')
            method_all_unique_authors = set()
            for i in utils.tqdm(range(len(method_authors)),desc='Create the unique authors file'):
                for author in method_authors[i]:
                    method_all_unique_authors.add(author)
            authors_node = dict(zip(method_all_unique_authors,
                                     [i for i in range(len(method_all_unique_authors))]))
            with open(utils._FILE_PATH_AUTHORS_NODE_BINARY, 'wb') as file:
                file.write(utils.pickle.dumps(authors_node))
            with open(utils.FILE_PATH_AUTHORS_NODE, 'w', encoding="utf-8") as file:
                for author, node in authors_node.items():
                    file.write(f"{author}:{node}\n")
    elif method=='first_letter_first_name':
        try:
            method_authors = \
                    utils.load_authors(utils.FILE_PATH_AUTHORS_FIRST_LETTER_FIRST_NAME,method)
            authors_node=utils.load_authors_node(utils.FILE_PATH_AUTHORS_NODE_FIRST_LETTER_FIRST_NAME,method)
        except:
            authors = utils.load_authors(utils.FILE_PATH_AUTHORS,'classic')
            method_all_unique_authors = set() # set with all authors of all papers
            method_authors = dict() #same as authors but with the new representation of the name
            for i in utils.tqdm(range(len(authors)),desc='Create the unique authors file'):
                method_authors_name = list()
                for autor in authors[i]:
                    split = autor.split()
                    name = split[-1:]
                    name.insert(0,split[0][0])
                    name = '.'.join(name)
                    method_authors_name.append(name)
                    method_all_unique_authors.add(name)
                method_authors[i] = method_authors_name
            authors_node = dict(zip(method_all_unique_authors,
                                     [i for i in range(len(method_all_unique_authors))]))
            with open(utils._FILE_PATH_AUTHORS_FIRST_LETTER_FIRST_NAME_BINARY, 'wb') as file:
                file.write(utils.pickle.dumps(method_authors))
            with open(utils.FILE_PATH_AUTHORS_FIRST_LETTER_FIRST_NAME, 'w', encoding="utf-8") as file:
                for papers_k, authors_i in method_authors.items():
                    file.write(f"{papers_k}|--|{','.join(authors_i)}\n")
            with open(utils._FILE_PATH_AUTHORS_NODE_FIRST_LETTER_FIRST_NAME_BINARY, 'wb') as file:
                file.write(utils.pickle.dumps(authors_node))
            with open(utils.FILE_PATH_AUTHORS_NODE_FIRST_LETTER_FIRST_NAME, 'w', encoding="utf-8") as file:
                for author, node in authors_node.items():
                    file.write(f"{author}:{node}\n")
    elif method=='unique':
        try:
            method_authors = \
                    utils.load_authors(utils.FILE_PATH_AUTHORS_UNIQUE,method)
            authors_node=utils.load_authors_node(utils.FILE_PATH_AUTHORS_NODE_UNIQUE,method)
        except:
            method_authors = utils.load_authors(utils.FILE_PATH_AUTHORS_UNIQUE,'unique')
            method_all_unique_authors = set()
            for i in utils.tqdm(range(len(method_authors)),desc='Create the unique authors file'):
                for author in method_authors[i]:
                    method_all_unique_authors.add(author)
            authors_node = dict(zip(method_all_unique_authors,
                                     [i for i in range(len(method_all_unique_authors))]))
            with open(utils._FILE_PATH_AUTHORS_NODE_UNIQUE_BINARY, 'wb') as file:
                file.write(utils.pickle.dumps(authors_node))
            with open(utils.FILE_PATH_AUTHORS_NODE_UNIQUE, 'w', encoding="utf-8") as file:
                for author, node in authors_node.items():
                    file.write(f"{author}:{node}\n")
    else:
        creation = False
        authors_node = dict()
        print("Method {} not implemented".format(method))
    
    return (authors_node, method_authors,creation)


def save_edges(i,array,file):
    """
    

    Parameters
    ----------
    i : int
        node i.
    array : np.array
        array contenning the weigth between author with node i and all authors with node > i.
    file : string
        file name.

    Returns
    -------
    None.

    """
    with open(file, 'a') as f:
        for m in range(len(array)):
            if array[m]>0: # if an edga between node i and m exist
                f.writelines(','.join([str(i),str(m+1+i),str(array[m])]))
                f.write('\n')

def write_txt_from_npz(csr_matrix,file_path_to_write):
    array1,array2 = csr_matrix.nonzero()
    with open(file_path_to_write, 'w', encoding="utf-8") as f:
        for i in range(len(array2)):
                f.writelines(','.join([str(array1[i]),str(array2[i]),str(csr_matrix[array1[i],array2[i]])]))
                f.write('\n')
        
def get_co_authors_graph(method,weighted):
    authors_node, method_authors,creation=create_files_co_authors_graph(method)
    #node_authors = {value:key for key,value in authors_node.items()}
    if method=='classic':
        file_path_save_graph= utils.FILE_PATH_CO_AUTHORS_GRAPH
    elif  method=='first_letter_first_name':
        file_path_save_graph = utils.FILE_PATH_CO_AUTHORS_GRAPH_FIRST_LETTER_FIRST_NAME
    elif method=='unique':
        file_path_save_graph = utils.FILE_PATH_CO_AUTHORS_GRAPH_UNIQUE
    else:
        print("Method {} not implemented. Be carfull with our results".format(method))
    try:
        graph = utils.load_weighted_graph(file_path_save_graph,weighted)
    except:
        if creation:
            #deep_copy_method_authors = copy.deepcopy(method_authors)
            #values_dict_method_authors = deep_copy_method_authors.values()
            values_dict_method_authors = method_authors.values()
            nb_of_autors = len(authors_node)
            authors_weights = lil_matrix((nb_of_autors,nb_of_autors),dtype=np.uintc)
            for autors_paper in utils.tqdm(values_dict_method_authors,desc='Creation of the co_authors_graph'):
                combi_temp = list(itertools.combinations(autors_paper,r=2))
                for elem in combi_temp:
                    node_1,node_2 = min(authors_node[elem[0]],authors_node[elem[1]]),max(authors_node[elem[0]],authors_node[elem[1]])
                    authors_weights[node_1,node_2] += 1
            csr_mat = csr_matrix(authors_weights)
            write_txt_from_npz(csr_mat,file_path_save_graph)
            graph = utils.load_weighted_graph(file_path_save_graph,weighted)
    return graph,authors_node, method_authors
# ajouter le cas où il eciste deja et return graph, authors_node, method_authors

def get__graph__authors_node__method_authors(method,weighted):
    return get_co_authors_graph(method,weighted)
    
def get_co_authors_embedding(file_name,graph,parameters,fast_mode,weighted=True):
    if fast_mode:
        try:
            n2v_co_authors = KeyedVectors.load_word2vec_format(file_name)
        except:
            fast_node2vec(graph,parameters,weighted,file_name)
            n2v_co_authors = KeyedVectors.load_word2vec_format(file_name)
    else:
        try:
            n2v_co_authors = np.load(file_name)
        except:
            n2v_co_authors = node2vec(graph,parameters,weighted,file_name)
        
    return n2v_co_authors

def get_papers_feature_from_co_authors_graph(method,parameters,fast_mode,weighted,concatenation_func=np.mean):
    graph, authors_node, method_authors = get__graph__authors_node__method_authors(method,weighted)
    weight_string =''
    if weighted:
        weight_string = 'weighted'
    
    if fast_mode:
        param_string = '_{}_{}_wl{}_d{}_context{}_q{}_p{}_epoch{}'.format(
            method,
            weight_string,
            parameters['walk_length'],
            parameters['dimensions'], 
            parameters['context'],
            parameters['q'],
            parameters['p'],
            parameters['epochs'])
        
        param_concat_string = '_{}_{}_{}_wl{}_d{}_context{}_q{}_p{}_epoch{}.npy'.format(
            re.search('(?<=function )(.*)(?= at)', str(concatenation_func)).group(),
            method,
            weight_string,
            parameters['walk_length'],
            parameters['dimensions'], 
            parameters['context'],
            parameters['q'],
            parameters['p'],
            parameters['epochs'])
        file_path_co_authors_embedding = utils.FILE_PATH_EMBEDDING_FAST_NODE2VEC_CO_AUTHORS + param_string
        file_path_papers_feature_from_co_authors_graph_embedding = \
            utils.FILE_PATH_EMBEDDING_FAST_NODE2VEC_PAPERS_FEATURE_FROM_CO_AUTHORS + param_concat_string
    else:
        param_string = '_{}_{}_wn{:d}_wl{:d}_d{:d}_ws{:d}.npy'.format(
                                                            method,
                                                            weight_string,
                                                            parameters['walk_number'], 
                                                            parameters['walk_length'],
                                                            parameters['dimensions'], 
                                                            parameters['window_size'])
        param_concat_string = '_'+str(re.search('(?<=function )(.*)(?= at)', str(concatenation_func)).group())+param_string
        
        file_path_co_authors_embedding=utils.FILE_PATH_EMBEDDING_NODE2VEC_CO_AUTHORS + param_string
        file_path_papers_feature_from_co_authors_graph_embedding=\
            utils.FILE_PATH_EMBEDDING_NODE2VEC_PAPERS_FEATURE_FROM_CO_AUTHORS + param_concat_string
    try:
        papers_feature = np.load(file_path_papers_feature_from_co_authors_graph_embedding)
    except:
        n2v_co_authors = get_co_authors_embedding(file_path_co_authors_embedding,graph,parameters,fast_mode,weighted)
        #print(len(n2v_co_authors),len(authors_node))
        '''
        zeros_len = len(n2v_co_authors[0])
        papers_feature = np.zeros((len(method_authors),zeros_len))
        for node,authors in method_authors.items():
            temp_list=list() # list contening embedding of all authors of one paper
            for author in authors:
                temp_list.append(n2v_co_authors[str(authors_node[author])])
            papers_feature[node] = concatenation_func(temp_list,axis=0)
        np.save(file_path_papers_feature_from_co_authors_graph_embedding,papers_feature)
        '''
    return n2v_co_authors
'''
fast_mode =False
weighted = True
parameters = {'walk_number': 10, 'walk_length': 15, 'dimensions': 64, 'window_size': 5, 'workers': 5}
n2v_co_authors = get_papers_feature_from_co_authors_graph('unique',
                                         parameters,
                                         fast_mode,
                                         weighted
                                         )
''' 

graph, authors_node, method_authors = get__graph__authors_node__method_authors('classic',True)
n2v_co_authors = get_co_authors_embedding('data/embedding_n2v_citation_graph_train_wn10_wl15_d64_ws5.npy',None,None,False,None)
zeros_len = len(n2v_co_authors[0])
papers_feature = np.zeros((len(method_authors),zeros_len))
for node,authors in method_authors.items():
    temp_list=list() # list contening embedding of all authors of one paper
    for author in authors:
        temp_list.append(n2v_co_authors[authors_node[author]])
    papers_feature[node] = np.max(temp_list,axis=0)
np.save('data/max_papers_features_from_embedding_n2v_citation_graph_train_wn10_wl15_d64_ws5.npy',papers_feature)
