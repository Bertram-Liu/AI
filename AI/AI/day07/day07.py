import tensorflow as tf
import os

# 关闭警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("job_name", " ", "启动服务的类型ps or worker")
tf.app.flags.DEFINE_integer("task_index", 0, "指定ps或者worker当中的哪一台服务器以task:0, task:1")


def main(argv):
    global_step = tf.contrib.framework.get_or_create_global_step()
    # 定义全部计数的op,  给钩子列表当中的训练步数使用

    # 1. 指定集群描述对象   ps, worker ps: 远程IP
    cluster = tf.train.ClusterSpec({"ps": ["172.40.80.198:2223"], "worker": ["172.40.80.200:2222"]})

    # 2. 创建不同的服务
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # 3. 根据不同的服务做不同的事情  ps: 更新保存参数  worker: 指定设备运行模型计算
    if FLAGS.job_name == "ps":
        # 参数服务器等待worker传递参数
        server.join()
    else:

        worker_device = "/job:worker/task:0/cpu:0"

        # 可以指定设备去运行
        with tf.device(tf.train.replica_device_setter(
                worker_device=worker_device, cluster=cluster
        )):
            # 简单做矩阵乘法运算
            x = tf.Variable([[1, 2, 3, 4]])
            w = tf.Variable([[2], [2], [3], [4]])

            mat = tf.matmul(x, w)

        # 创建分布式会话
        with tf.train.MonitoredTrainingSession(
                master="grpc://172.40.80.200:2222",  # 指定主worker
                is_chief=(FLAGS.task_index == 0),  # 判断是否是主worker
                config=tf.ConfigProto(log_device_placement=True),  # 打印设备信息
                hooks=[tf.train.StopAtStepHook(last_step=200)]  # 钩子函数
        ) as mon_sess:
            # 如果mon_sess没有报异常
            while not mon_sess.should_stop():
                print(mon_sess.run(mat))

        return argv


if __name__ == '__main__':
    # 默认调用main函数
    tf.app.run()
