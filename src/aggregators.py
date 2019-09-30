import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, session, train_data, labels, item_indices, user_indices
                 , act=tf.nn.relu, dropout=0., name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        self.labels = labels
        self.item_indices = item_indices
        self.user_indices = user_indices
        self.train_data = train_data
        self.session = session

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        self.session.run(tf.global_variables_initializer())

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        """
        合并neighbor_vectors、neighbor_relations和user_embeddings三个张量，
        将此结果和self_vectors张量相加，dropout正则化后，乘以权重矩阵，加上偏移量，最后输出用户对每个邻居电影的分数矩阵
        :param self_vectors: 电影随机向量，来源于电影索引矩阵
        :param neighbor_vectors: 用户邻居向量，来源于实体-实体关联列表
        :param neighbor_relations: 用户关系向量，来源于实体-关系关联列表
        :param user_embeddings: 用户嵌入向量，来源于用户矩阵
        :param masks: 无用
        :return: 根据电影向量，求出的用户对邻居的分数矩阵
        """
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        # 用户对邻居的分数向量 + 电影本身的随机向量
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        """
        点乘用户随机向量和关系向量，对每一行(每一个邻居的关系)求平均值后再求交叉熵正则化，得到用户关系评分向量；
        再将用户关系评分向量点乘邻居张量，对每一列(每一个维度)求均值后，做为合并的结果输出

        对于每一个用户而言：
            先求他对每一个邻居的关系的分数，
            再计算他对每一个邻居的分数，
            最后计算每一个维度(共32个)下，他对16个邻居的平均值

        :param neighbor_vectors: 邻居张量，来源于实体-实体关联列表
        :param neighbor_relations: 关系张量，来源于实体-关系关联列表
        :param user_embeddings: 用户嵌入向量张量，来源于用户矩阵
        :return: 每个用户在每个维度下，对16个邻居的平均分，[65536 * 1 * 32]
        """
        # [batch_size, 1, 1, dim]
        user_embeddings_reshape = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

        # [batch_size, -1, n_neighbor]
        user_relation_scores = tf.reduce_mean(user_embeddings_reshape * neighbor_relations, axis=-1)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

        # [batch_size, -1, dim]
        neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)

        return neighbors_aggregated

    def test_tensor(self, tensor):
        return self.session.run([tensor], self.get_feed_dict_test())

    def get_feed_dict_test(self):
        start = 0
        end = self.batch_size
        data = self.train_data

        feed_dict = {self.user_indices: data[start:end, 0],
                     self.item_indices: data[start:end, 1],
                     self.labels: data[start:end, 2]}
        return feed_dict


class LabelAggregator(Aggregator):
    def __init__(self, batch_size, dim,
                 session, train_data, labels, item_indices, user_indices, name=None):
        super(LabelAggregator, self).__init__(batch_size, dim, 0., None, name)

        self.labels = labels
        self.item_indices = item_indices
        self.user_indices = user_indices
        self.train_data = train_data
        self.session = session

    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        """
        实体标签扩展(propagation)
        :param self_labels: 元素全为0.5
        :param neighbor_labels: 元素全为0.5
        :param neighbor_relations: 实体-关系向量
        :param user_embeddings: 用户向量
        :param masks: 元素全为0
        :return: 实体标签 list，[1, 65536]
        """
        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

        # [batch_size, -1, n_neighbor]
        user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1]
        neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_labels, axis=-1)
        # neighbor_labels元素全为0.5，相当于把用户-关系分数/2，再取每一行的均值
        output = tf.cast(masks, tf.float32) * self_labels + tf.cast(
            tf.logical_not(masks), tf.float32) * neighbors_aggregated  # 实际就是neighbors_aggregated

        return output

    def test_tensor(self, tensor):
        return self.session.run([tensor], self.get_feed_dict_test())

    def get_feed_dict_test(self):
        start = 0
        end = self.batch_size
        data = self.train_data

        feed_dict = {self.user_indices: data[start:end, 0],
                     self.item_indices: data[start:end, 1],
                     self.labels: data[start:end, 2]}
        return feed_dict
