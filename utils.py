# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:10:15 2022

@author: sacha
"""

import os
from tqdm import tqdm
import pickle
import errno
import networkx as nx


###############################         ###############################
############################### AUTHORS ###############################
#### paper's authors information but with classic or first_letter_first_name representation
FILE_PATH_AUTHORS = 'data/authors.txt'
_FILE_PATH_AUTHORS_BINARY = 'data/authors_preprocessed.pkl'

FILE_PATH_AUTHORS_FIRST_LETTER_FIRST_NAME = 'data/first_letter_first_name_authors.txt'
_FILE_PATH_AUTHORS_FIRST_LETTER_FIRST_NAME_BINARY = 'data/first_letter_first_name_authors_preprocessed.pkl'

FILE_PATH_AUTHORS_UNIQUE = 'data/unique_authors.txt'
_FILE_PATH_AUTHORS_UNIQUE_BINARY = 'data/unique_authors_preprocessed.pkl'

#### author's node information
FILE_PATH_AUTHORS_NODE = 'data/authors_node.txt'
_FILE_PATH_AUTHORS_NODE_BINARY = 'data/authors_node_preprocessed.pkl'

FILE_PATH_AUTHORS_NODE_FIRST_LETTER_FIRST_NAME = 'data/first_letter_first_name_authors_node.txt'
_FILE_PATH_AUTHORS_NODE_FIRST_LETTER_FIRST_NAME_BINARY = 'data/first_letter_first_name_authors_node_preprocessed.pkl'

FILE_PATH_AUTHORS_NODE_UNIQUE = 'data/unique_authors_node.txt'
_FILE_PATH_AUTHORS_NODE_UNIQUE_BINARY = 'data/unique_authors_node_preprocessed.pkl'
##### graph of co_autor

FILE_PATH_CO_AUTHORS_GRAPH = 'data/co_authors_graph.txt'
FILE_PATH_CO_AUTHORS_GRAPH_FIRST_LETTER_FIRST_NAME = 'data/first_letter_first_name_co_authors_graph.txt'
FILE_PATH_CO_AUTHORS_GRAPH_UNIQUE = 'data/unique_co_authors_graph.txt'

##### Embedding format

FILE_PATH_EMBEDDING_FAST_NODE2VEC_CO_AUTHORS = 'data/embedding_fast_node2vec_co_authors'
FILE_PATH_EMBEDDING_FAST_NODE2VEC_PAPERS_FEATURE_FROM_CO_AUTHORS ='data/embedding_fast_node2vec_papers_feature_co_authors'
FILE_PATH_EMBEDDING_NODE2VEC_CO_AUTHORS = 'data/embedding_node2vec_co_authors'
FILE_PATH_EMBEDDING_NODE2VEC_PAPERS_FEATURE_FROM_CO_AUTHORS ='data/embedding_node2vec_papers_feature_co_authors'

########################################################################
#########################################################################

FILE_PATH_EDGE_LIST = 'data/edgelist.txt'
FILE_PATH_EDGE_LIST_VALIDATION = 'data/edgelist_val.txt'
FILE_PATH_EMBEDDING_FAST_NODE2VEC_EDGE_LIST = 'data/embedding_fast_node2vec_edgelist'

def load_authors(file_path_authors,method):
    current_path = os.getcwd()
    authors = dict()
    if os.path.isfile(current_path+'/'+ file_path_authors):
        if method=='classic':
            file_path_authors_binary = _FILE_PATH_AUTHORS_BINARY
        elif  method=='first_letter_first_name':
            file_path_authors_binary = _FILE_PATH_AUTHORS_FIRST_LETTER_FIRST_NAME_BINARY
        elif method=='unique':
            file_path_authors_binary = _FILE_PATH_AUTHORS_UNIQUE_BINARY
        else:
            print("Method {} not implemented. Be carfull with our results \n".format(method))
        try:
            a_file = open(file_path_authors_binary, "rb")
            authors = pickle.load(a_file)
            a_file.close()
            print('Authors are load from : ' + current_path+'/'+ file_path_authors_binary+'\n')
        except:
            with open(file_path_authors, 'r', encoding="utf-8") as f:
                for line in tqdm(f,desc='Loading file'):
                    node, author = line.split('|--|')
                    author = author.split(',')
                    author[-1] = author[-1].strip()
                    authors[int(node)] = author
                a_file = open(file_path_authors_binary, "wb")
                pickle.dump(authors, a_file)
                a_file.close()
            print('Authors are load from : ' + current_path+'/'+ file_path_authors+'\n')
    else:
        raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT),  current_path+'\\'+file_path_authors)
        
    return authors

def load_authors_node(file_path_authors_node,method):
    current_path = os.getcwd()
    authors_node = dict()
    if os.path.isfile(current_path+'/'+ file_path_authors_node):
        if method=='classic':
            file_path_authors_binary = _FILE_PATH_AUTHORS_NODE_BINARY
        elif  method=='first_letter_first_name':
            file_path_authors_binary = _FILE_PATH_AUTHORS_NODE_FIRST_LETTER_FIRST_NAME_BINARY
        elif method == 'unique':
            file_path_authors_binary = _FILE_PATH_AUTHORS_NODE_UNIQUE_BINARY
        else:
            print("Method {} not implemented. Be carfull with our results".format(method))
            
        try:
            a_file = open(file_path_authors_binary, "rb")
            authors_node = pickle.load(a_file)
            a_file.close()
            print('Authors node are load from : ' + current_path+'/'+ file_path_authors_binary+'\n')
        except:
            with open(file_path_authors_node, 'r', encoding="utf-8") as f:
                for line in tqdm(f,desc='Loading file'):
                    node, author = line.split(':')
                    authors_node[int(node)] = author
            print('Authors node are load from : ' + current_path+'/'+ file_path_authors_node+'\n')
                    
    else:
        raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), current_path+'/'+file_path_authors_node+'\n')
    
    return authors_node

def load_weighted_graph(file_path_graph,weighted=True):
    if not weighted:
        G = nx.read_edgelist(file_path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
    else:
        G = nx.read_weighted_edgelist(file_path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
        
    return G