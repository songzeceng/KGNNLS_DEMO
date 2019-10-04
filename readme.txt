1、目前是根据movieLens数据集运行代码的，预处理阶段已经完成，可以直接运行main.py来执行训练和测试。
main.py直接调用的是train.py文件中的train()方法，主要逻辑有四：构造用户-电影交互哈希表，模型构造、训练和评估、topk(推荐k部电影)

2、构造用户-电影交互哈希表对应函数get_interaction_table()，
就是把用户id和电影id进行整合，做为键，用户电影是否有过交互，做为值(0:没交互、1:有交互)

3、模型构造的主要逻辑都在model.py文件中KGNN_LS类的构造方法里，主要分为build_model和build_train两步。
前者构造模型，后者计算损失函数，并进行梯度优化

3.1、构造模型又分为两步，aggregate和_build_label_smoothness_loss
前者主要是根据知识图谱中Item(电影)矩阵、实体矩阵(电影或属性)和实体之间的关系矩阵，以及用户向量，
计算出用户对某个电影的16个邻居的分数，构成一个用户-邻居分数矩阵。
这部分代码主要集中在aggregators.py中的SumAggregator._call()方法里

后者主要是根据实体矩阵、关系矩阵和用户-电影交互矩阵，计算电影的标签，做为预测值，后面对其进行逻辑回归。
计算电影的预测标签主要集中在aggregators.py中的LabelAggregator._call()方法里

3.2、计算损失函数时，总的损失函数分为几部分，公式如下：
total_loss = base_loss + label_smooth_weight * label_smooth_loss + l2_weight * l2_loss
其中，base_loss是对分数矩阵(用户矩阵点乘用户-邻居分数矩阵)求逻辑回归和平均交叉熵损失的结果
label_smooth_loss是对电影的真实标签和预测标签求逻辑回归和平均交叉熵损失的结果，label_smooth_weight是其权重
l2_loss是四部分的l2损失的和，这四部分是：邻居矩阵、实体-实体矩阵、实体-关系矩阵和aggregate时的权重矩阵，l2_weight是l2_loss的权重
(l2就是岭回归，可有效缓解过拟合问题，参见博文https://blog.csdn.net/wfei101/article/details/80767096)

最后用Adam优化器(一种梯度下降优化器)去最小化total_loss(代码中是loss)

4、模型训练就是给模型的占位符填充数据，然后训练Adam优化器去最小化total_loss。对应model.py中KGNN_LS类的train()方法
填充的数据为：训练集第一列填充用户id向量(用于构造用户矩阵)、第二列填充电影id向量(用于构造电影矩阵)、第三列填充真实标签向量
填充数据对应train.py的get_feed_dict()方法

5、模型评估对应train.py的ctr_eval()方法，分别对训练集、测试集和评估集进行auc和f1评估
具体的评估过程对应model.py中KGNN_LS类的eval()方法。
其中auc分数，用于二分类的评估。分数越大，分类器越好，f1分数就是2/(准确率倒数 + 召回率倒数)
(此处参见博文https://blog.csdn.net/guhongpiaoyi/article/details/53289229)

6、topk是可选过程，因为它的准确率和召回率都不高。
主要是先用整个测试集，对每个用户计算分数矩阵，
然后把分数矩阵排序，获取前k行(也就是前k个电影)，
最后根据测试集中用户看过的电影数，去求这次推荐的准确率和召回率

候选k值有[1, 2, 5, 10, 20, 50, 100]

结果为：
precision: 0.2100	0.1750	0.1200	0.0990	0.0765	0.0524	0.0394
recall: 0.0286	0.0510	0.0814	0.1542	0.2167	0.3191	0.4518


ps：为了方便在断点调试时能够查看张量的值，我把会话对象和一些填充数据对象，
做为了模型类和Aggregator类的属性，然后定义了test_tensor()方法。
断点调试时，直接在evaluate窗口运行test_tensor()就行，传入参数就是要测试的张量对象
