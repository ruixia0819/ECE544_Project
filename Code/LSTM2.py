from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from dataset import DataSet
import matplotlib.pyplot as plt



# Parameters
learning_rate = 0.001
training_iters = 1000000
# training_iters = 1000
batch_size = 500
display_step = 5

# Network Parameters
n_input = 100  # MNIST data input (img shape: 28*28)
n_steps = 60  # timesteps
n_hidden = 100  # hidden layer num of features
# n_hidden = n_steps
n_classes = 6  # MNIST total classes (0-9 digits)

# datasets
train_data_set = DataSet(path='./data_set/train', max_length=n_steps)
# eval_data_set = DataSet(path='./data_set/eval', max_length=n_steps)
test_data_set = DataSet(path='./data_set/test', max_length=n_steps)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    ## try
    line_out = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.nn.softmax(line_out)

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph

Accuracy_=[]
iteration=[]


with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations

    # gen = sentences.__iter__()



    print('start')
    while step * batch_size < training_iters:
        batch_x, batch_y = train_data_set.next_batch_stupid(batch_size)
        # batch_x, batch_y = train_data_set.next_batch_stupid_shuffle(batch_size)
        # batch_x, batch_y = train_data_set.next_batch(batch_size)

        # if step % 1000000000 == 0.1:
        #     print(batch_x.shape)
        # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape((batch_size, 60, 100))
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss

            Accuracy_.append(acc)
            iteration.append(step)

            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    # test_data_set = DataSet(path='./data_set/test')

    # Calculate accuracy for 128 mnist test images
    test_len = 1000
    test_data, test_label = test_data_set.next_batch_stupid_shuffle(test_len)
    # test_data, test_label = mnist.test.images.reshape((-1, 60, 100))
    # test_data = test_data[:, ::4, ::4]
    # test_data = test_data[:test_len]
    test_data = test_data.reshape((-1, n_steps, n_input))

    # test_label = test_label[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    plt.title('Iteration time & batch size')
    plt.plot(iteration, Accuracy_, linewidth=2.5, linestyle='-')
    plt.savefig('./fig/%s_%s_%s_%s.png' % (n_input, n_hidden, learning_rate, training_iters))

print(iteration)
print(Accuracy_)
plt.show()
