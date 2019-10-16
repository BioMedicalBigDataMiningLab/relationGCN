import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act #激活函数
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            # sparse_tensor_dense_matmul(
            #     sp_a,
            #     b,
            #     adjoint_a=False,
            #     adjoint_b=False,
            #     name=None
            # ) 用稠密矩阵“B”乘以 SparseTensor(秩为 2)“A”
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x) #激活函数
        return outputs

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')  #初始化权重
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            #将矩阵 a 乘以矩阵 b,生成a * b
            x = tf.sparse_tensor_dense_matmul(self.adj, x) #相乘

            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            x = tf.transpose(inputs)
            x = tf.matmul(inputs, x)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs

def weight_variable_glorot(input_dim, output_dim, name=""):  #正态初始化
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    # random_uniform(
    #     shape,
    #     minval=0,
    #     maxval=None,
    #     dtype=tf.float32,
    #     seed=None,
    #     name=None
    # )
    return tf.Variable(initial, name=name)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape) #从均匀分布中输出随机值.
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool) #tf.floor返回不大于 x 的元素最大整数
    pre_out = tf.sparse_retain(x, dropout_mask) #在一个 SparseTensor 中保留指定的非空值.
    return pre_out * (1. / keep_prob)