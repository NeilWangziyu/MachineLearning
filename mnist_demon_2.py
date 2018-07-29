import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取数据（如果存在就读取，不存在就下载完再读取）
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入
x = tf.placeholder("float", [None, 784]) #输入占位符（每张手写数字784个像素点）
y_ = tf.placeholder("float", [None,10]) #输入占位符（这张手写数字具体代表的值，0-9对应矩阵的10个位置）

# 计算分类softmax会将xW+b分成10类，对应0-9
W = tf.Variable(tf.zeros([784,10])) #权重
b = tf.Variable(tf.zeros([10])) #偏置
y = tf.nn.softmax(tf.matmul(x,W) + b) # 输入矩阵x与权重矩阵W相乘，加上偏置矩阵b，然后求softmax（sigmoid函数升级版，可以分成多类）

# 计算偏差和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用梯度下降法（步长0.01），来使偏差和最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10): # 训练10次
  batch_xs, batch_ys = mnist.train.next_batch(100) # 随机取100个手写数字图片
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 执行梯度下降算法，输入值x：batch_xs，输入值y：batch_ys

# 计算训练精度
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #运行精度图，x和y_从测试手写图片中取值