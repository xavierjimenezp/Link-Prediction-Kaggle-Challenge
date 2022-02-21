# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:16:10 2022

@author: sacha
"""

from test_phase import create_files_co_authors_graph,get_co_authors_embedding
from utils import load_weighted_graph
from tqdm import tqdm
import numpy as np
from scipy.sparse import lil_matrix,csr_matrix,save_npz, load_npz
import itertools
import networkx as nx
#from FastNode2Vec_embedding import fast_node2vec

def read_edge_list():
    list_tot = list()
    with open('data/edgelist_train.txt', 'r', encoding="utf-8") as f:
        for line in tqdm(f,desc='Loading file'):
            edge1,edge2 = line.split(',')
            list_tot.append([int(edge1),int(edge2)])
        #a_file = open(file_path_authors_binary, "wb")
        #pickle.dump(authors, a_file)
        #a_file.close()
    return list_tot
def save_edges_version_pas_plus_1(i,array,file):
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
                f.writelines(','.join([str(i),str(m+i),str(array[m])]))
                f.write('\n')
    
def main():
    authors_node, method_authors,_ =create_files_co_authors_graph('classic')
    nb_authors = len(authors_node)
    edgelist = read_edge_list()
    ## prepo
    paper_authors_node = dict()
    edgelist_author_id = list()
    for paper, authors in method_authors.items():
        authors_node_temp = [authors_node[author_temp] for author_temp in authors]
        paper_authors_node[paper] = authors_node_temp # int faster than string for comparaison
    for papers in edgelist:
        paper1 = paper_authors_node[papers[0]]
        paper2 = paper_authors_node[papers[1]]
        edgelist_author_id.append([paper1,paper2])
    count = 0
    authors_citation_number = lil_matrix((nb_authors,nb_authors),dtype=np.uintc)
    for papers in tqdm(edgelist_author_id):
        # will updat only lower triangular element with diag
        product =[x for x in list(itertools.product(*papers)) if x[0]<=x[1]]
        for couple_authors in product:
            authors_citation_number[couple_authors[0],couple_authors[1]] +=1
            count+=1
    save_npz("data/yourmatrix_train.npz", csr_matrix(authors_citation_number))
    return count

#print(main())
def create_txt(file):
    a = load_npz(file)
    file_graph = 'data/citation_graph_train.txt'
    with open(file_graph, 'w', encoding="utf-8") as f:
        array1,array2 = a.nonzero()
        for i in range(len(array2)):
            f.writelines(','.join([str(array1[i]),str(array2[i]),str(a[array1[i],array2[i]])]))
            f.write('\n')
    G = nx.read_weighted_edgelist(file_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
    numeric_indices = [index for index in range(G.number_of_nodes())]
    node_indices = sorted([node for node in G.nodes()])
    
    if not numeric_indices==node_indices:
        with open(file_graph, 'a', encoding="utf-8") as f:
            max_nodes = max(G.nodes())
            missing_list = list(set([index for index in range(max_nodes)]) - set(node_indices))
            print("The node indexing is wrong. But the graph was complete by node not connected\n")
            for i in tqdm(missing_list,desc='Waiting for completion'):
                f.writelines(','.join([str(i),str(i),str(0)]))
                f.write('\n')
                

         
     
main()   
create_txt("data/yourmatrix_train.npz")
'''
method = 'classic'
weight_string = 'weighted'
parameters = {'walk_length': 80, 'dimensions': 64,'context':10,'p':1.0,'q':1.0,'epochs':5, 'workers': 7}
file_path_citation_embedding = 'data/embedding_fast_node2vec_citation_{}_{}_wl{}_d{}_context{}_q{}_p{}_epoch{}'.format(
        method,
        weight_string,
        parameters['walk_length'],
        parameters['dimensions'], 
        parameters['context'],
        parameters['q'],
        parameters['p'],
        parameters['epochs'])
graph = load_weighted_graph('data/citation_graph.txt')
emb = get_co_authors_embedding(graph,parameters,True,file_path_citation_embedding)


      '''      
        
    
    
    