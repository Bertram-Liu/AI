from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


# 关闭警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 获取真实的数据
# mnist = input_data.read_data_sets("./mnist/", one_hot=True)

# print(mnist)
# print(mnist.train.images)
# print("************")
# print(mnist.train.labels)
# print("****")
# print(mnist.train.images[0])

# 批次获取数据
# print(mnist.train.next_batch(50))

# 定义命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("is_train", 1, "指定程序是预测还是训练")


def full_connect():
    """
    单层 (全连接层) 实现手写数字识别
    :return: None
    """
    # 获取训练数据
    mnist = input_data.read_data_sets("./mnist/", one_hot=True)

    # 1. 建立数据占位符  特征值x [None, 784], 目标值y_true [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])  # 特征值
        y_true = tf.placeholder(tf.int32, [None, 10])  # 真实值

    # 2. 建立全连接层的神经网络 w [784, 10] b [10]
    with tf.variable_scope("fc_model"):
        # 随机初始化权重和偏置 (x * w) + b
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="w")
        bias = tf.Variable(tf.constant(0.0, shape=[10]))  # shape: 1维

        # 预测None个样本的输出结果 矩阵matrix相乘 [None, 784] * [784, 10] + [10] = [None, 10]
        y_predict = tf.matmul(x, weight) + bias

    # 3. 计算所有样本的损失 (平均值)
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 反向传播 (梯度下降优化)
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # learning_rate: [value] minimize: [loss_min]

    # 5. 计算准确率: accuracy_train
    with tf.variable_scope("acc_train"):
        # 预测准确置为1, 否则置为0
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))
        # 转换float32类型, 对所有值求平均值
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 观察损失率和准确率的变化
    # 1. 收集变量:  ---单个数字值
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    #    高纬度收集:
    tf.summary.histogram("weights", weight)
    tf.summary.histogram("biases", bias)

    # 定义初始化变量op   *****
    init_op = tf.global_variables_initializer()

    # 定义一个合并变量op
    merged = tf.summary.merge_all()

    # 创建一个saver
    saver = tf.train.Saver()

    # 开启会话训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 建立evens文件, 然后写入
        file_writer = tf.summary.FileWriter("./tmp/summary/test", graph=sess.graph)

        if FLAGS.is_train == 1:
            # 迭代步数训练, 更新参数预测
            for i in range(2000):
                # 提供训练数据  批次: 50
                # 取出真实存在的特征值和目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)

                # 运行train_op训练
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                # 写入每部训练的值
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                file_writer.add_summary(summary, i)

                print("训练第%d步, 准确率为:%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

            # 保存模型
            saver.save(sess, "./tmp/ckpt/fc_model")  # 起名字: [fc_model]
        else:
            # 加载模型
            saver.restore(sess, "./tmp/ckpt/fc_model")

            # 如果是0, 做出预测100张图片
            for _ in range(100):
                # 每次测试1张图片  x_test, y_test: one_hot编码
                x_test, y_test = mnist.test.next_batch(1)
                print("第%d张图片, 手写数字图片目标是:%d, 预测结果是:%d" % (
                    _,
                    tf.arg_max(y_test, 1).eval(),
                    tf.arg_max(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                ))

    return None


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    """
    自定义的卷积模型
    :return: None
    """
    # 1. 准备数据的占位符 x[None, 784] y_true[None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])  # 特征值
        y_true = tf.placeholder(tf.int32, [None, 10])  # 真实值

    # 2. 一卷积层: 卷积, 激活, 池化
    with tf.variable_scope("conv1"):
        # 1) 卷积: 5*5*1, 32个Filter, strides=1
        # 随机初始化权重, 偏置
        w_conv1 = weight_variables([5, 5, 1, 32])
        b_conv1 = bias_variables([32])  # Filter个数: 32

        # 对x进行形状的改变 [None, 784] >>> [None, 28, 28, 1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])  # 改变形状不知道时填-1

        # 2) 2-D_tensor卷积 + Relu激活
        #       strides(统一): 传递中间两个值[1, 1, 1, 1]
        #       [None, 28, 28, 1] >>> [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        # 3) 池化: 2*2, strides=2, [None, 28, 28, 32] >>> [None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3. 二卷积层
    with tf.variable_scope("conv2"):
        # 1) 卷积: 5*5*32, 64个Filter, strides=1
        # 随机初始化权重, 偏置
        w_conv2 = weight_variables([5, 5, 32, 64])
        b_conv2 = bias_variables([64])

        # 2) 2-D_tensor卷积 + Relu激活
        #       strides(统一): 传递中间两个值[1, 1, 1, 1]
        #       [None, 14, 14, 32] >>> [None, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # 3) 池化: 2*2, strides=2, [None, 14, 14, 64] >>> [None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4. 全连接层: [None, 7*7*64] * [7*7*64, 10] + [10] = [None, 10]
    with tf.variable_scope("fc"):
        # 随机初始化权重, 偏置
        w_fc = weight_variables([7 * 7 * 64, 10])
        b_fc = bias_variables([10])

        # 修改形状 [None, 7, 7, 64] >>> [7*7*64, 10]
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])  # 改变形状不知道时填-1

        # 进行矩阵运算得出每个样本的10个结果
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def conv_fc():
    """
    卷积层实现手写数字
    :return: None
    """
    # 1. 获取训练数据
    mnist = input_data.read_data_sets("./mnist/", one_hot=True)

    # 2. 定义模型, 得出输出
    x, y_true, y_predict = model()

    # 进行交叉熵损失计算
    # 3. 计算所有样本的损失 (平均值)
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 反向传播 (梯度下降优化)
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(
            loss)  # learning_rate: [value] minimize: [loss_min]

    # 5. 计算准确率: accuracy_train
    with tf.variable_scope("acc_train"):
        # 预测准确置为1, 否则置为0
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))
        # 转换float32类型, 对所有值求平均值
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 开启会话运行
    with tf.Session() as sess:
        # 初始化op
        sess.run(init_op)

        # 循环训练
        for i in range(1000):
            # 提供训练数据  批次: 50
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 运行train_op训练
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            print("训练第%d步, 准确率为:%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

    return None


if __name__ == '__main__':
    conv_fc()
