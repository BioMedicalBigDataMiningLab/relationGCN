from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp

import networkx as nx
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from deep.model import GCNModel
from deep.optimizer import Optimizer

from tensorflow.python import debug as tfdbg  # 调试

# Set random seed
seed = 200
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings

flags = tf.app.flags
tf.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()  # 转换存储格式，稀疏矩阵
    coords = np.vstack((sparse_mx.row,
                        sparse_mx.col)).transpose()  # https://blog.csdn.net/csdn15698845876/article/details/73380803 #位置

    values = sparse_mx.data  # 值
    shape = sparse_mx.shape  # 形状
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)  # 稀疏化存储
    adj_ = adj + sp.eye(adj.shape[0])  # 加上对角线上的连接
    rowsum = np.array(adj_.sum(1))  # 每行相加
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 对角线为度矩阵对角线开平方分之一
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})  # 空的单位矩阵的另外一种存储 feed_dict是给placeholder提供值

    feed_dict.update({placeholders['adj']: adj_normalized})  # 跟度矩阵处理后的值
    feed_dict.update({placeholders['adj_orig']: adj})  # 邻近矩阵
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 2% positive links
    # Remove diagonal elements #adj邻近矩阵
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()  # 得到不含自循环的临近矩阵

    adj_triu = sp.triu(adj)  # 得到adj的上三角矩阵
    adj_tuple = sparse_to_tuple(adj_triu)  # 上三角矩阵转换成元组
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]  # 位置信息
    # num_test = int(np.floor(edges.shape[0] / 50.))
    # num_val = int(np.floor(edges.shape[0] / 50.))

    num_test = int(np.floor(edges.shape[0] / 50.))  # 测试集数量
    num_val = int(np.floor(edges.shape[0] / 50.))  # 验证集数量

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)  # 打乱
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]  # 测试的边
    val_edges = edges[val_edge_idx]  # 验证的边
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)  # 训练的边

    # import numpy as np
    # a = [1, 2, 3]
    # b = [4, 5, 6]
    # print(np.hstack((a, b)))
    #
    # 输出：[1 2 3 4 5 6]

    # import numpy as np
    # a = [[1], [2], [3]]
    # b = [[1], [2], [3]]
    # c = [[1], [2], [3]]
    # d = [[1], [2], [3]]
    # print(np.hstack((a, b, c, d)))
    #
    # 输出：
    # [[1 1 1 1]
    #  [2 2 2 2]
    # [3 3 3 3]]
    # np.hstack水平(按列顺序)把数组给堆叠起来

    def ismember(a, b):  # a是否在b中
        rows_close = np.all((a - b[:, None]) == 0, axis=-1)
        # python 的广播
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        n_rnd = len(test_edges) - len(test_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)  # 随机产生2*n_rnd个从0到adj.shape[0]的数
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        n_rnd = len(val_edges) - len(val_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])  # array([1., 1., 1., 1., 1.])
    # 生成3*1的全是的矩阵
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # train_edges[:, 0], train_edges[:, 1] 分别是第一列和第二列
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


#得到新的打分矩阵
def get_new_scoring_matrices(adj):
    # adj = np.loadtxt(open(csvfile, "rb"), delimiter=",", skiprows=0)
    adj=sp.csr_matrix(adj) #转换为稀疏矩阵
    num_nodes = adj.shape[0]  # 两类节点不加区分，总共微生物292个，疾病38个，加起来 330个
    num_edges = adj.sum()
    # Featureless
    features = sparse_to_tuple(sp.identity(num_nodes))  # 输入
    num_features = features[2][1]  # 列数
    features_nonzero = features[1].shape[0]  # data的形状，也还是节点数

    # Store original adjacency matrix (without diagonal entries) for later
    # np.newaxis None,None增加维度https://blog.csdn.net/qq_36490364/article/details/83594271
    adj_orig = adj - sp.dia_matrix((adj.diagonal(), [0]), shape=adj.shape)  # todo  把临近矩阵的对角线1元素去掉
    # dia_matrix((data, offsets), shape=(M, N)) 对角线无偏移

    adj_orig.eliminate_zeros()  # 不存储值为0的元素

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj = adj_train

    adj_norm = preprocess_graph(adj)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # tf.sparse_placeholder()为稀疏张量插入占位符,该稀疏张量将始终被提供.

    # Create model
    model = GCNModel(placeholders, num_features, features_nonzero, name='yeast_gcn')

    # Create optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    # Initialize session
    sess = tf.Session()
    # sess = tfdbg.LocalCLIDebugWrapperSession(sess)  # 被调试器封装的会话
    # sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())

    adj_label = adj+ sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        # features 是三个值，第一个有值的位置信息，第二个值，第三个形状信息
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        # feed_dict.update({placeholders['features']: features})  # 空的单位矩阵的另外一种存储 feed_dict是给placeholder提供值
        #
        # feed_dict.update({placeholders['adj']: adj_norma})  # 跟度矩阵处理后的值
        # feed_dict.update({placeholders['adj_orig']: adj})  # 邻近矩阵
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # One update of parameter matrices
        # print(sess.run(feed_dict))
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        # Performance on validation set
        # def get_roc_score(edges_pos, edges_neg):函数 定义


    print('Optimization Finished!')
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    adj_rec = sigmoid(np.dot(emb, emb.T))
    np.savetxt('output.csv',adj_rec,delimiter=',')
    return adj_rec



