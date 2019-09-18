# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
#import torch
import numpy  as np
import scipy.sparse as sp

def onehot_encode(labes):
    classes = set(labes)
    #classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    classes_dict = {c: i for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get,labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

def load_data(path="./data/",dataset="cora"):
    print("load {} dataset ..... ".format(dataset))
    idx_fea_label = np.genfromtxt("{}{}/cora.content".format(path,dataset),dtype=np.dtype(str))
    #print(idx_fea_label.shape)
    idx = idx_fea_label[:,0]
    idx_map = {j: i for i, j in enumerate(idx)}
    #features = idx_fea_label[:,1:-1]
    features = sp.csr_matrix(idx_fea_label[:, 1:-1], dtype=np.float32)
    labels_or = idx_fea_label[:,-1]
    labels = onehot_encode(labels_or)
    edge_orig = np.genfromtxt("{}{}/cora.cites".format(path,dataset),dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edge_orig.flatten())),dtype=np.int32).reshape(edge_orig.shape)
    #print(edge_orig)
    #print(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print(type(adj))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    # print(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
#load_data()
if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    print(features.shape)
    print(set(labels))
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    print(classes_dict)