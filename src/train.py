import tensorflow as tf
import numpy as np
from model import KGNN_LS


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    init = (tf.global_variables_initializer(), tf.local_variables_initializer())

    result = str()

    with open("../result.txt", encoding="utf-8", mode="w") as f:
        f.write("")
        f.flush()

    with tf.Session() as sess:
        sess.run(init)

        interaction_table, offset = get_interaction_table(train_data, n_entity)
        interaction_table.init.run()

        model = KGNN_LS(args, n_user, n_entity, n_relation,
                        adj_entity, adj_relation, interaction_table,
                        offset, sess, train_data)

        # top-K evaluation settings
        user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    result += ("epoch %d, start %d, loss %.4f\n" % (step, start, loss))
                    print(start, loss)

            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

            result += ('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f\n'
                       % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))
            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f\n'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                result += 'precision: \n'
                print('precision: ', end='')
                for i in precision:
                    result += ('%.4f\t' % i)
                    print('%.4f\t' % i, end='')
                print()
                result += '\n'
                result += 'recall: \n'
                print('recall: ', end='')
                for i in recall:
                    result += ('%.4f\t' % i)
                    print('%.4f\t' % i, end='')
                result += '\n'
                print('\n')

            with open("../result.txt", encoding="utf-8", mode="a") as f:
                f.write(result)
                f.flush()


# interaction_table is used for fetching user-item interaction label in LS regularization
# key: user_id * 10^offset + item_id
# value: y_{user_id, item_id}
def get_interaction_table(train_data, n_entity):
    '''
    构造用户-电影交互情况的哈希表
    :param train_data: 训练集，内容为[userId, movieId, 是否有过交互(0-无，1-有)
    :param n_entity: 实体数量
    :return: 用户-电影交互哈希表，[userId * n + movieId, 0 or 1]，偏移量n(10 ^ len(n_entity))
    '''
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)

    interaction_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item):
    """
    从数据集中读取每个用户看过的电影，并随机抽取100个用户返回
    :param show_topk: 是否启用topk功能
    :param train_data: 训练集
    :param test_data: 测试集
    :param n_item: 电影数量
    :return: 随机的100个用户列表、用户-电影训练记录、用户-电影测试记录、电影id集合、k_list
    """
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    """
    求出数据集的auc分数和f1分数的均值，以评估预测结果

    :param sess: 会话对象
    :param model: KGNN-LS模型对象
    :param data: 数据集(训练、测试、评估)
    :param batch_size: 批处理规模
    :return: 此数据集的auc分数和f1分数的均值
    """
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    """
    评估topk
    :param sess:会话对象
    :param model:KGNN-LS模型对象
    :param user_list:随机用户列表
    :param train_record:训练集中，用户对应的电影列表，字典
    :param test_record:测试集中，用户看过的电影列表，字典
    :param item_set:电影id集合
    :param k_list:不同的k值列表
    :param batch_size:批处理数量
    :return:不同k值的准确率召回率均值
    """
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        # while start + batch_size <= len(test_item_list):
        #     items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
        #                                             model.item_indices: test_item_list[start:start + batch_size]})
        #     for item, score in zip(items, scores):
        #         item_score_map[item] = score
        #     start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            # 用每个user做为user_indices、测试数据集(不够一批次的，用最后一个元素填满)填充KGNN-LS模型
            # 然后获取item_indices和用户电影评分向量
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        # 把分数从高到低排序，并获取相应的电影id
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            # 针对每一种k值，求其准确率和召回率。
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    # 每一种k值的准确率召回率均值
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    """
    获取全部训练集数据和测试集中用户已看过的电影
    :param data: 数据集
    :param is_train: 数据集是否是训练集
    :return: 每个用户对应的训练集中全部电影和测试集中用户看过的电影，字典。
    """
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
