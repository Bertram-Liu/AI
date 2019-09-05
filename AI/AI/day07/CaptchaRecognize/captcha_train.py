import tensorflow as tf
import os

# 关闭警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS

# 定义命令行参数
tf.app.flags.DEFINE_string("captcha_dir", "./tfrecords/captcha.tfrecords", "验证码数据的路径")
tf.app.flags.DEFINE_integer("batch_size", 100, "每批次训练的样本数")
tf.app.flags.DEFINE_integer("label_num", 4, "每个样本的目标值数量")
tf.app.flags.DEFINE_integer("letter_num", 26, "每个目标值取的字母的可能性个数")


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def read_and_decode():
    """
    读取数据API
    :return: image_batch, label_batch
    """
    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer([FLAGS.captcha_dir])

    # 2. 构建阅读器, 读取文件内容 (默认一个样本)
    reader = tf.TFRecordReader()

    #    读取内容
    key, value = reader.read(file_queue)

    # 3. 解析Example
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),  # 形状不设定
        "label": tf.FixedLenFeature([], tf.string),
    })

    # 解码内容, 字符串内容
    #   1) 解析图片的特征值
    image = tf.decode_raw(features["image"], tf.uint8)

    #   2) 解析图片的目标值
    label = tf.decode_raw(features["label"], tf.uint8)

    # print(image, label)

    # 4. 改变形状
    image_reshape = tf.reshape(image, [20, 80, 3])
    label_reshape = tf.reshape(label, [4])

    print(image_reshape, label_reshape)

    # 5. 进行批处理, 每次读取的样本数 100
    image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], batch_size=FLAGS.batch_size,
                                              num_threads=1, capacity=FLAGS.batch_size)

    print(image_batch, label_batch)

    return image_batch, label_batch


def fc_model(image):
    """
    进行预测结果
    :param image: 100张图片特征值 [100, 20, 80, 3]
    :return: y_predict预测值 [100, 4*26]
    """
    with tf.variable_scope("model"):
        # 将图片数据形状转换成二维的形状
        image_reshape = tf.reshape(image, [-1, 20 * 80 * 3])

        # 1. 随机初始化权重, 偏置
        # matrix[100, 20 * 80 * 3] * [20 * 80 * 3, 4 * 26] + [4 * 26] = [100, 4 * 26]
        weights = weight_variables([20 * 80 * 3, 4 * 26])
        bias = bias_variables([4 * 26])

        # 进行全连接层计算
        y_predict = tf.matmul(tf.cast(image_reshape, tf.float32), weights) + bias

    return y_predict


def predict_to_onehot(label):
    """
    将读取的文件当中的目标值转换成one_hot编码
    :param label: [100, 4] >>>  [[13, 25, 15, 15], [19, 23, 20, 16]]
    :return: one_hot
    """
    # 进行one_hot编码转换, 提供给交叉熵损失计算, 准确率计算
    label_onehot = tf.one_hot(label, depth=FLAGS.letter_num, on_value=1.0, axis=2)  # 3-D: axis=2

    print(label_onehot)

    return label_onehot


def captcharec():
    """
    验证码识别程序
    :return: None
    """
    # 1. 读取验证码的数据文件 label_batch [100, 4]
    image_batch, label_batch = read_and_decode()

    # 2. 通过输入图片特征数据, 建立模型, 得出预测结果
    # 一层, 全连接神经网络进行预测
    # matrix [100, 20*80*3] * [20*80*3, 4*26] + [4*26] = [100, 4*26]
    y_predict = fc_model(image_batch)

    print(y_predict)

    # 3. 先把目标值转换成one_hot编码 [100, 4, 26]
    y_true = predict_to_onehot(label_batch)

    # 4. sofamax计算, 交叉熵损失计算,
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失, y_true [100, 4, 26] >>> [100,  4*26]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(y_true, [FLAGS.batch_size, FLAGS.label_num * FLAGS.letter_num]),
            logits=y_predict))

    # 5. 梯度下降优化损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 6. 计算准确率          :三维比较
    with tf.variable_scope("acc"):
        # 比较预测值和目标值是否位置(4)一样    y_predict [100, 4*26] >>> [100, 4, 26]
        equal_list = tf.equal(tf.arg_max(y_true, 2),
                              tf.arg_max(tf.reshape(y_predict, [FLAGS.batch_size, FLAGS.label_num, FLAGS.letter_num]),
                                         2))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义初始化变量op   *****
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 初始化变量op
        sess.run(init_op)

        # 定义线程协调器和开启线程 (有数据在文件当中读取提供给模型)
        coord = tf.train.Coordinator()

        # 开启线程运行读取文件操作
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 训练识别程序
        for i in range(5000):
            sess.run(train_op)

            print("第%d批次的准确率为: %f" % (i, accuracy.eval()))

        # 回收线程
        coord.request_stop()
        coord.join(threads)

    return image_batch, label_batch


if __name__ == '__main__':
    captcharec()
