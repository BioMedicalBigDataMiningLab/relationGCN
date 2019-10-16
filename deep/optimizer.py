import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

class Optimizer():
    def __init__(self, preds, labels, num_nodes, num_edges):
        pos_weight = float(num_nodes ** 2 - num_edges) / num_edges
        norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)

        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)