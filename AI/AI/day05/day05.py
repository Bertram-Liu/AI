import tensorflow as tf
import os

# 关闭警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 模拟一下同步先处理数据, 然后才能取数据训练
# tensorflow当中, 运行操作有依赖性

# # 1. 首先定义队列
# Q = tf.FIFOQueue(3, tf.float32)
#
# # 放入一些数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])  # 防止混淆 + ,
#
# # 2. 定义一些处理数据的逻辑, 取数据的过程       取数据, +1, 入队列
# out_q = Q.dequeue()
# data = out_q + 1
# en_q = Q.enqueue(data)
#
# with tf.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)
#
#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)
#
#         # 训练数据
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


# # 模拟异步子线程 存入样本, 主线程 读取样本
# # 1. 定义一个队列, 1000
# Q = tf.FIFOQueue(1000, tf.float32)
#
# # 2. 定义子线程要做的事情, 值, +1, 放入队列当中
# var = tf.Variable(0.0)
#
# # 实现一个自增
# data = tf.assign_add(var, tf.constant(1.0))
# en_q = Q.enqueue(data)
#
# # 3. 定义队列管理器op, 指定多少个子线程该干什么事情
# qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
#
# # 初始化变量的op
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 初始化op
#     sess.run(init_op)
#
#     # 开启线程管理器
#     coord = tf.train.Coordinator()
#
#     # 真正开启子线程
#     threads = qr.create_threads(sess, coord=coord, start=True)
#
#     # 主线程, 不断读取数据
#     for i in range(300):
#         print(sess.run(Q.dequeue()))
#
#     # 请求其他线程终止
#     coord.request_stop()
#     # 关闭线程
#     coord.join(threads)


def csvread(filelist):
    """
    读取csv文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    # 1. 构造文件队列  shuffle: 默认是True(有序) False(无序)
    file_queue = tf.train.string_input_producer(filelist, shuffle=True)

    # 2. 构造csv阅读器读取队列  (按一行)
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    # print(key, value)

    # 3. 解码: 对每行内容解码
    # record_defaults: 指定每一个样本的每一列的类型, 指定默认值 [["None"], [2.0]]
    records = [["None"], ["None"]]

    # 返回每一列的值  example: 第一列   label: 第二列
    example, label = tf.decode_csv(value, record_defaults=records)

    # 4. 读取多个数据, 批处理   capacity: 队列大小
    # 批处理大小, 与队列, 数据的数量没有影响, 取决于多少数据 batch_size=20: 循环取数据
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)
    print(example_batch, label_batch)

    return example_batch, label_batch


def picread(filelist):
    """
    读取狗图片并转换张量
    :param filelist: 文件路径+ 名字的列表
    :return: 每张图片的张量
    """
    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2. 构造阅读器读取图片内容 (默认读取一张图片)
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)
    print(value)

    # 3. 解码: 对读取的图片数据进行解码
    image = tf.image.decode_jpeg(value)
    print(image)

    # 4. 处理图片大小  (统一像素)
    image_resize = tf.image.resize_images(image, [200, 200])
    print(image_resize)

    # 注意: 一定要把样本的形状固定 [200, 200, 3]
    image_resize.set_shape([200, 200, 3])

    # 5. 进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)
    print(image_batch)

    return image_resize


# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_dir", "./cifar10/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./temp/cifar.tfrecords", "存进的tfrecords文件")


class CifarRead(object):
    """
    完成读取二进制文件, 写进tfrecords, 读取tfrecods
    """

    def __init__(self, filelist):
        # 定义文件列表
        self.file_list = filelist

        # 定义读取图片的属性
        self.height = 32
        self.width = 32
        self.channel = 3  # 通道数

        # 二进制文件的每张图片的字节
        self.label_bytes = 1  # 标签的字节
        self.image_bytes = self.height * self.width * self.channel  # 图片的字节
        self.bytes = self.label_bytes + self.image_bytes  # 每张图片的字节

    # 读取二进制文件, 转换tensor
    def read_and_decode(self):
        # 1. 构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2. 构造二进制文件阅读器, 读取内容, 每个样本的字节数
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read(file_queue)
        print(value)

        # 3. 解码内容
        label_image = tf.decode_raw(value, tf.uint8)
        print(label_image)

        # 4. 分割图片和标签数据  特征值和目标值
        # tf.cast 转换数据类型
        label = tf.cast(tf.slice(label_image, [0], [1]), tf.int32)
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        # 5. 对图片的特征数据进行形状的改变 [3072] >>> [32, 32, 3]    :维度的改变
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        print(label, image_reshape)

        # 6. 批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=20, num_threads=1, capacity=20)
        print(image_batch, label_batch)

        return image_batch, label_batch

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值和默认值存进tfrecords
        :param image_batch: 10张图片的特征值
        :param label_batch: 10张图片的目标值
        :return: None
        """
        # 1. 构造tfrecords文件
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 循环将所有样本写入文件, 每张图片样本都要构造Example协议块
        for i in range(10):
            # 取出第i个图片的特征值和目标值
            image = image_batch[i].eval().tostring()
            label = label_batch[i].eval()[0]  # 取出值

            # 构造一个样本的Example
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # value: 字符串
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  # value: 值
            }))

            # 写入单独的样本  序列化成json格式
            writer.write(example.SerializeToString())

        # 关闭
        writer.close()

        return None

    def read_from_tfrecords(self):
        """
        读取tfrecords文件内容
        :return: None
        """
        # 1. 构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        # 2. 构造文件阅读器, 读取内容example, value: 一个样本的序列化example
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        # 3. 解析example
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),  # shapw: [], dtype: tf.string
            "label": tf.FixedLenFeature([], tf.int64),
        })
        print(features["image"], features["label"])

        # 4. 解码内容  if读取的内容是string需要解码, elif: int64, float32 不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)
        label = tf.cast(features["label"], tf.int32)
        print(image, label)

        # 5. 统一图片大小
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        print(image_reshape, label)

        # 6. 进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)
        # print(image_batch, label_batch)

        return image_batch, label_batch


if __name__ == '__main__':
    # 找到文件, 放入列表    路径+名字 >>> 列表当中
    # file_name = os.listdir("./csvdata/")
    # file_name = os.listdir("./dog/")
    file_name = os.listdir(FLAGS.cifar_dir)

    # print(filename)  # 返回文件名列表

    # 拼接文件名
    # filelist = [os.path.join("./csvdata/", file) for file in file_name]
    # filelist = [os.path.join("./dog/", file) for file in file_name]
    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]  # 取出.bin文件

    # example_batch, label_batch = csvread(filelist)
    # example_batch, label_batch = picread(filelist)

    # image_batch = CifarRead(filelist)

    cf = CifarRead(filelist)
    # image_batch, label_batch = cf.read_and_decode()

    image_batch, label_batch = cf.read_from_tfrecords()

    # 开启回话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 存进tfrecords文件
        # print("开始存储")
        #
        # cf.write_to_tfrecords(image_batch, label_batch)
        #
        # print("结束存储")

        # 循环取3次 乱序
        # for i in range(3):

        # 打印读取的内容
        # print(sess.run([example_batch, label_batch]))
        # print(sess.run([image_batch]))
        print(sess.run([image_batch, label_batch]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)
