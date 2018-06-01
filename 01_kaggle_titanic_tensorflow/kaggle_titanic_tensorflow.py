import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./train.csv') # 891 rows x 12 columns
test_data = pd.read_csv('./test.csv') # 418 rows x 11 columns

# 이 부분은 데이터 전처리~~~
# pandas에서는 축의 이름을 따라 데이터를 정렬할 수 있는 자료구조인 DataFrame을 사용!!
# ---------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import Imputer

# 데이터의 NaN값들을 채워주기 위한 함수
def nan_padding(data, columns):
    for column in columns:
        # Imputer는 missing value들을 채워주는 scikit-learn의 클래스
        imputer=Imputer()
        # fit_transform 메소드는 데이터 fit과 transform을 같이 해줌
        # fit=train이라고 보면 되고, transform은 실제로 데이터를 변화시키는 것
        # default가 mean이기 때문에 해당 column의 평균값으로 NaN을 채움
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

# NaN을 가지는 column
nan_columns = ["Age", "SibSp", "Parch"]

train_data = nan_padding(train_data, nan_columns)
test_data = nan_padding(test_data, nan_columns)

# ---------------------------------------------------------------------------------------------------------------------

# 코드 마지막 부분 evaluation을 위해 PassengerId를 저장~
test_passenger_id=test_data["PassengerId"]

# ---------------------------------------------------------------------------------------------------------------------

# 생존율과 관련없는 column을 없애는 함수
def drop_not_concerned(data, columns):
    # pandas 메소드 drop 사용
    return data.drop(columns, axis=1)

# 생존율과 관련없는 column
not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]

train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)

# pandas 메소드 head는 데이터의 첫 n행을 return(default=5)
print(train_data.head())
print(test_data.head())

# ---------------------------------------------------------------------------------------------------------------------

# categorical data를 one-hot encoding해주는 함수
def dummy_data(data, columns):
    for column in columns:
        # get_dummies는 categorical data를 one-hot encoding해주는 pandas 함수
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        # 뒤쪽에 concat해줬으니 기존에 있는 것 없애줘야함
        data = data.drop(column, axis=1)
    return data

# categorical data를 가지는 column
dummy_columns = ["Pclass"]

train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)
print(train_data.head())

# ---------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

# 성별을 integer로 바꿔주는 함수
def sex_to_int(data):
    # LabelEncoder는 label을 0과 n_classes-1사이의 값으로 encoding해주는 scikit-learn 클래스
    le = LabelEncoder()
    # fit 메소드를 사용해 label 입력
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"])
    return data

train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)
print(train_data.head())

# ---------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

# 나이를 0에서 1 사이의 값으로 변환해주는 함수
def normalize_age(data):
    # scikit-learn의 MinMaxScaler 클래스 사용
    scaler = MinMaxScaler()
    # default range가 (0, 1)
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data

train_data = normalize_age(train_data)
test_data = normalize_age(test_data)
print(train_data.head())

# ---------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# 데이터를 "Survived" column과 나머지 column으로 나누고, 또 이를 train 데이터와 validation 데이터로 나누는 함수
def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    # label을 이진법 변환해주는 scikit-learn 클래스인데 왜 썼지?
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    # scikit-learn의 train_test_split 함수 사용해 비율대로 쪼갬
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

print("valid_x:{}".format(valid_x.shape))
print("valid_y:{}".format(valid_y.shape))

# numpy array로 변환되어서 이제 head 못 씀
print(train_x[:5])
print(train_y[:5])

# ---------------------------------------------------------------------------------------------------------------------

# 여기서부터 tensorflow 딥러닝인데 알아서 보자~~~~~

from collections import namedtuple

def build_neural_network(hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.Variable(True, dtype=tf.bool)

    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None, kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    fc = tf.nn.relu(fc)

    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate', 'is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

model = build_neural_network()

def get_batch(data_x, data_y, batch_size=32):
    batch_n = len(data_x) // batch_size
    for i in range(batch_n):
        batch_x = data_x[i * batch_size:(i + 1) * batch_size]
        batch_y = data_y[i * batch_size:(i + 1) * batch_size]

        yield batch_x, batch_y

epochs = 200
train_collect = 50
train_print = train_collect * 2

learning_rate_value = 0.001
batch_size = 16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

# 체크포인트 저장을 위해 Saver 사용
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 0
    for e in range(epochs):
        for batch_x, batch_y in get_batch(train_x, train_y, batch_size):
            iteration += 1
            feed = {model.inputs: train_x,
                    model.labels: train_y,
                    model.learning_rate: learning_rate_value,
                    model.is_training: True
                    }

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)

            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print == 0:
                    print("[Epoch: {}/{}]".format(e + 1, epochs),
                          "Train Loss: {:.4f}".format(train_loss),
                          "Train Acc: {:.4f}".format(train_acc))

                feed = {model.inputs: valid_x,
                        model.labels: valid_y,
                        model.is_training: False
                        }
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)

                if iteration % train_print == 0:
                    print("[Epoch: {}/{}]".format(e + 1, epochs),
                          "Validation Loss: {:.4f}".format(val_loss),
                          "Validation Acc: {:.4f}".format(val_acc))

    saver.save(sess, "./checkpoint/titanic.ckpt")

# ---------------------------------------------------------------------------------------------------------------------

# matplotlib으로 loss랑 accuracy 그래프 찍어보기

plt.plot(x_collect, train_loss_collect, "r--")
plt.plot(x_collect, valid_loss_collect, "g^")
plt.show()

plt.plot(x_collect, train_acc_collect, "r--")
plt.plot(x_collect, valid_acc_collect, "g^")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------

# 체크포인트 불러와서 테스트 데이터 10개 predict 해보기

model = build_neural_network()
restorer = tf.train.Saver()
with tf.Session() as sess:
    restorer.restore(sess, "./checkpoint/titanic.ckpt")
    feed = {
        model.inputs: test_data,
        model.is_training: False
    }
    test_predict = sess.run(model.predicted, feed_dict=feed)

print(test_predict[:10])

# Binarizer로 0,1로 바꿔서 생존 예측

from sklearn.preprocessing import Binarizer

binarizer=Binarizer(0.5)
test_predict_result=binarizer.fit_transform(test_predict)
test_predict_result=test_predict_result.astype(np.int32)

print(test_predict_result[:10])

# ---------------------------------------------------------------------------------------------------------------------

# 앞서 저장해둔 PassengerId로 DataFrame을 만들고 생존 예측한 것을 column으로 추가
# csv파일로 저장

passenger_id=test_passenger_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=test_predict_result

print(evaluation[:10])

evaluation.to_csv("evaluation_submission.csv",index=False)