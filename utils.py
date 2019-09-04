import os
import torch
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance
import ABIDEParser as Reader

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, float(-1)).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_labels():
    
    subject_IDs = Reader.get_ids()                                   
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP') 
    labels = list(map(int, list(labels.values()))) 
    labels = np.array(labels) - 1                                   
    return labels

feature_names = ['SEX', 'SITE_ID', 'AGE_AT_SCAN']
feat_length = len(feature_names)

def get_phenot_vector(subj_id):

    sites = np.array(['PITT', 'OLIN', 'OHSU', 'SDSU', 'TRINITY', 'UM_1', 'UM_2', 'USM',
       'YALE', 'CMU', 'LEUVEN_1', 'LEUVEN_2', 'KKI', 'NYU', 'STANFORD',
       'UCLA_1', 'UCLA_2', 'MAX_MUN', 'CALTECH', 'SBL'])

    root_folder = os.path.dirname(os.path.abspath(__file__))
    phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
    
    df = pd.read_csv(phenotype)
    df.set_index("subject", inplace=True)
    features = df.loc[int(subj_id), feature_names].tolist()
    site_int_id = (np.where(sites == features[1]))[0][0]
    features[1] = site_int_id + 1
    return np.array(features)

def get_all_phenot_vectors():
    
    subject_IDs = Reader.get_ids() 
    phenot_X = np.zeros((len(subject_IDs), feat_length))
    
    idx = 0
    for subj_id in subject_IDs:
        features = get_phenot_vector(subj_id)
        phenot_X[idx] = features
        idx += 1 
    return phenot_X

def create_weighted_adjacency():
    
    phenot_X = get_all_phenot_vectors()
    Y = distance.pdist(phenot_X, 'hamming')*3
    Y = 3 - distance.squareform(Y)
    return Y

def get_num_edges():

    Y = create_weighted_adjacency()
    up_idx = np.triu_indices(len(Y))
    upper_tri_matrix = Y[up_idx]
    num_edges = len(upper_tri_matrix[upper_tri_matrix>0]) - len(Y)
    return num_edges

def load_ABIDE(graph_type): 

    atlas = 'ho'
    connectivity = 'correlation'
   
    # Get class labels
    subject_IDs = Reader.get_ids()                                   
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP') 
    
    # Get acquisition site
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')   
    unique = np.unique(list(sites.values())).tolist()               
    
    num_classes = 2
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    site = np.zeros([num_nodes, 1], dtype=np.int)

    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1              
        site[i] = unique.index(sites[subject_IDs[i]])
    
    # Compute feature vectors (vectorised connectivity networks)
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)
    
    labels = list(map(int, list(labels.values()))) 
    labels = np.array(labels) - 1                                   

    # Compute population graph using phenotypic features
    if graph_type == 'original':
        final_graph = create_weighted_adjacency() 
    if graph_type == 'graph_no_features':
        final_graph = create_weighted_adjacency() 
        features = np.identity(num_nodes)
    if graph_type == 'graph_random':
        ones = get_num_edges()/(len(labels)*len(labels))
        final_graph = np.random.choice([0, 1], size=(len(labels), len(labels)), p = [1 - ones, ones])
        final_graph = (final_graph + final_graph.T)/2 
    if graph_type == 'graph_identity':
        final_graph = np.zeros((num_nodes, num_nodes))
        
    final_graph = normalize(final_graph) 
    
    adj = sp.coo_matrix(final_graph)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    
    features = sp.csr_matrix(features)
    features = normalize(features)
    
    # Convert to tensors
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense())).float()
    
    return adj, features, labels    