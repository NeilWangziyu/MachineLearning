import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# make some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# print(x_data)
# print(x_data.shape)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#add hidden layer
l1 = add_layer(xs, 1, 10,activation_function=tf.nn.relu)

# add out put layer
prediction = add_layer(l1,10,1,activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
# reduction_indices=[1] 结果压缩方向

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# import step
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1500):
    # training
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        # 图片中去除掉lines的第一个线段
        plt.pause(0.3)
