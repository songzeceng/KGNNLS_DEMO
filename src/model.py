import tensorflow as tf
from aggregators import SumAggregator, LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGNN_LS(object):
    def __init__(self, args, n_user, n_entity, n_relation,
                 adj_entity, adj_relation, interaction_table,
                 offset, session, train_data):
        self.session = session
        self.train_data = train_data
        self.initializer = KGNN_LS.get_initializer()
        # self.session.run(self.initializer)

        self._parse_args(args, adj_entity, adj_relation, interaction_table, offset)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

        self.session.run(tf.initialize_all_variables())

    @staticmethod
    def get_initializer():
        """
        Xavier初始化方法，好处是不管是前向反馈还是后向反馈，每一层的输出方差都相等
        :return: Xavier初始化张量
        """
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, interaction_table, offset):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        # LS regularization
        self.interaction_table = interaction_table
        self.offset = offset
        self.ls_weight = args.ls_weight

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=self.initializer, name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=self.initializer, name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=self.initializer, name='relation_emb_matrix')

        self.session.run(tf.global_variables_initializer())

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregator = self.aggregate(entities, relations)

        # LS regularization
        self._build_label_smoothness_loss(entities, relations)

        # [batch_size]
        # 用户向量点乘用户电影评分向量，再对每一行求和，做为用户电影评分向量，最后对其求sigmoid
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        """
        从实体-实体关联列表和实体-关系关联矩阵中取样本
        样本尺寸均为[65536(self.batch_size), 16(args.neighbor_sample_size)]
        :param seeds: self.item_indices张量，根据其中索引从实体和关系列表矩阵中取值
        :return: 取到的样本值
        """
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        # for i in range(self.n_iter):
        neighbor_entities = tf.reshape(tf.gather(self.adj_entity, seeds), [self.batch_size, -1])
        neighbor_relations = tf.reshape(tf.gather(self.adj_relation, seeds), [self.batch_size, -1])
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations):
        """
        根据实体和关系列表，将实体矩阵、关系矩阵、用户矩阵和电影索引张量合并
        :param entities: 实体列表
        :param relations: 关系列表
        :return: 用户电影关系分数矩阵(65536 * 32)，张量聚集器
        """
        '''
        此处从随机矩阵entity_emb_matrix和relation_emb_matrix中按索引取值
        对于实体随机矩阵：
            第一次的索引是电影id
            第二次的索引是样本邻居实体的id
        对于关系随机矩阵：
            索引为关系id
        对于每一个矩阵，都把取到的样本保存到向量里
        
        embedding_lookup(矩阵，索引矩阵)，按照索引矩阵读取矩阵，
        如果索引矩阵有多行，则按其第一行的值，读取矩阵的对应行
        '''
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh,
                                   session=self.session, train_data=self.train_data,
                                   labels=self.labels, user_indices=self.user_indices,
                                   item_indices=self.item_indices)

        shape = [self.batch_size, -1, self.n_neighbor, self.dim]
        vector = aggregator(self_vectors=entity_vectors[0],
                            neighbor_vectors=tf.reshape(entity_vectors[1], shape),
                            neighbor_relations=tf.reshape(relation_vectors[0], shape),
                            user_embeddings=self.user_embeddings,
                            masks=None)

        res = tf.reshape(vector, [self.batch_size, self.dim])

        return res, aggregator

    # LS regularization
    def _build_label_smoothness_loss(self, entities, relations):
        """
        根据实体-实体关联向量、实体-关系关联向量、用户索引向量在交互表中的存在关系，生成实体标签(扩展)
        :param entities: 实体-实体关联向量
        :param relations: 实体-关系关联向量
        :return: 无
        """
        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        reset_masks = []  # True means the label of this item is reset to initial value during label propagation
        holdout_item_for_user = None

        for entities_per_iter in entities:
            # [batch_size, 1]
            users = tf.expand_dims(self.user_indices, 1)
            # [batch_size, n_neighbor^i]
            user_entity_concat = users * self.offset + entities_per_iter  # userId和itemId结合

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            # [batch_size, n_neighbor^i]
            initial_label = self.interaction_table.lookup(user_entity_concat)
            # 某个用户和电影集的交互情况
            # 对于电影知识图谱中的情况，全为0.5(即查不到)
            holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)
            # 对于用户电影交互情况：初始化holdout_mask，全为False
            # 对于电影知识图谱中邻居情况，初始化holdout_mask，全为True
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)
            # True则表示用户和这个电影发生过交互，False则表示没有
            # 对于电影知识图谱，reset_mask全为false

            reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # label initialization
            # 初始化initial_label，元素全为0.5

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration

        # label propagation
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        aggregator = LabelAggregator(self.batch_size, self.dim,
                                     session=self.session, train_data=self.train_data,
                                     labels=self.labels, user_indices=self.user_indices,
                                     item_indices=self.item_indices)

        vector = aggregator(self_vectors=entity_labels[0],
                            neighbor_vectors=tf.reshape(
                                entity_labels[1], [self.batch_size, -1, self.n_neighbor]),
                            neighbor_relations=tf.reshape(
                                relation_vectors[0], [self.batch_size, -1, self.n_neighbor, self.dim]),
                            user_embeddings=self.user_embeddings,
                            masks=reset_masks[0])

        self.predicted_labels = tf.squeeze(vector, axis=-1)
        # 把[1, 65536]的向量变成[65536, 1]的向量，元素值不变，做为预测的标签

    def _build_train(self):
        # base loss
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))  # 对用户电影评分向量求交叉熵损失，运用逻辑回归

        # L2 loss
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)  # 对用户向量、实体邻居向量和实体关系向量的l2范式损失求和
        # for aggregator in self.aggregators:
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.aggregator.weights)  # 加上聚集器里的权重
        self.loss = self.base_loss + self.l2_weight * self.l2_loss  # 默认l2权重是1e-7

        # LS loss
        self.ls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.predicted_labels))  # 对预测的标签求交叉熵损失，并运用逻辑回归
        self.loss += self.ls_weight * self.ls_loss  # 默认ls权重是1

        # loss = 用户电影评分的逻辑回归交叉熵损失 +
        #        l2权重 * l2损失(用户向量 + 实体邻居向量 + 实体关系向量) +
        #        预测标签的逻辑回归交叉熵损失

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        """
        用roc_auc对比labels(真实值)和scores_normalized(预测值)，评价分类结果
        :param sess: 会话对象
        :param feed_dict: 填充数据
        :return: auc分数和f1分数
        """
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        # roc_auc分数，用于二分类的评估。分数越大，分类器越好
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

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
