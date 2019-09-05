from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, silhouette_score
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mylinear():
    """
    线性回归预测房子价格
    :return: None
    """
    # 数据获取
    lb = load_boston()
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train)
    print(y_test)
    print(x_train.shape)
    print(y_train.shape)

    # 标准化处理 (特征值和目标值)
    # 特征值
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)  # 样本数不知道填-1, 目标值填1
    x_test = std_x.transform(x_test)
    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)
    # 预测房价结果
    model = joblib.load("./temp/test.pkl")
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print("保存的模型预测的结果: ", y_predict)

    # estimator 预测
    # 正规方程求解方式预测结果
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("回归系数: ", lr.coef_)
    # 预测测试集的房子价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    y_lr_predict = lr.predict(x_test)
    print("正规方程测试集里面每个房子的预测价格: ", y_lr_predict)
    print("正规方程的均方误差: ", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    # 梯度下降去进行房价预测
    sgd = SGDRegressor(eta0=0.02)
    sgd.fit(x_train, y_train)
    print("回归系数: ", sgd.coef_)

    # 预测测试集的房子价格
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    # y_predict = sgd.predict(x_test)

    print("梯度下降测试集里面每个房子的预测价格: ", y_sgd_predict)
    print("梯度下降的均方误差: ", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    # 岭回归去进行房价预测

    rd = Ridge(alpha=1.0)  # 0~1 1~10
    rd.fit(x_train, y_train)
    print("回归系数: ", rd.coef_)

    # 预测测试集的房子价格

    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    # y_predict = rd.predict(x_test)
    print("岭回归测试集里面每个房子的预测价格: ", y_rd_predict)
    print("岭回归的均方误差: ", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return None


def logistic():
    """
    逻辑回归做二分类进行癌症预测 (根据细胞的属性特征)
    :return: None
    """

    # 构造列标签名字
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
    # 读取数据
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column)
    print(data)

    # 缺失值进行处理
    data = data.replace(to_replace="?", value=np.nan)
    # 删除?
    data = data.dropna()

    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)

    y_predict = lg.predict(x_test)
    print(lg.coef_)
    print("准确率: ", lg.score(x_test, y_test))
    print("召回率: ", classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    return None


def km():
    """
    K-means进行聚类分析
    :return: None
    """

    # 读取四张表的数据
    prior = pd.read_csv("./prior.csv")
    products = pd.read_csv("./products.csv")
    orders = pd.read_csv("./orders.csv")
    aisles = pd.read_csv("./aisles.csv")

    # 合并四张表到一张表 (用户-物品类别)
    _mg = pd.merge(prior, products, on=['product_id', 'product_id'])
    _mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
    mt = pd.merge(_mg, aisles, on=['aisles_id', 'aisles_id'])

    print(mt.head(10))

    # 交叉表(特殊的分组工具)
    cross = pd.crosstab(mt['user_id'], ['aisle'])
    print(cross.head(10))

    # 进行主成分分析
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(cross)
    x = data[:500]
    print(x.shape)

    # 假设用户四个类别
    km = KMeans(n_clusters=4)
    km.fit(x)
    predict = km.predict(x)
    print(predict)

    # 显示聚类的结果
    plt.figure(figsize=(10, 10), dpi=80)

    # 建立四个颜色的列表
    colored = ['orange', 'green', 'blue', 'purple']
    colr = [colored[i] for i in predict]
    plt.scatter(x[:, 1], color=colr)
    plt.xlabel("1")
    plt.ylabel("20")
    plt.show()

    #  评判聚类效果, 轮廓系数
    silhouette_score(x, predict)
    return None


if __name__ == '__main__':
    km()
