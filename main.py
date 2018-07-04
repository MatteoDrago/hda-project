import numpy as np
import pandas as pd
import os
import tensorflow as tf


# column indexes of signals to be kept
idx_signals = list(np.arange(1,46)) \
            + list(np.arange(50,59)) \
            + list(np.arange(63,72)) \
            + list(np.arange(76,85)) \
            + list(np.arange(89,98)) \
            + list(np.arange(102,134))
# column indexes of labels
idx_labels = list(np.arange(243,250))
# import dataset
features_all = pd.read_table('./OpportunityUCIDataset/dataset/S1-ADL1.dat', sep="\s+", header=None, usecols=idx_signals)
labels_all = pd.read_table('./OpportunityUCIDataset/dataset/S1-ADL1.dat', sep="\s+", header=None, usecols=idx_labels)

print('\nImported data:\n\n', features_all.head())
print('\nImported data:\n\n', labels_all.head())

# interpolation
# os.system("interpolation.py")


# CLASSIFICATION

# parameters
batch_size, seq_length, n_channels = 1, 50, 113
stride = 25
activity_label = 1
labels = labels_all.iloc[:,activity_label]

print("\nBatch size: ", batch_size, "\nSequence length: ", seq_length)
print("\nLabels:\n", labels.head())

# placeholders
X = tf.placeholder(tf.float32, shape=[None, seq_length, n_channels], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1], name='label')

# layers
conv_1 = tf.layers.conv1d(inputs=X, filters=64, kernel_size=2, activation=tf.nn.relu)
max_pool_1 = tf.layers.max_pooling1d(inputs=conv_1, pool_size=2, strides=2, padding='same')
dropout_1 = tf.layers.dropout(inputs=max_pool_1, rate=0.3)

conv_2 = tf.layers.conv1d(inputs=dropout_1, filters=36, kernel_size=1, activation=tf.nn.relu)
max_pool_2 = tf.layers.max_pooling1d(inputs=conv_2, pool_size=2, strides=2, padding='same')
dropout_2 = tf.layers.dropout(inputs=max_pool_2, rate=0.3)

full_1 = tf.layers.dense(inputs=dropout_2, units=10)

y_pred = tf.layers.dense(inputs=full_1, units=4)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))

# optimizer
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# session
def next_batch(step, seq_length, batch_size):
    idx_from = step * stride
    batch_x = features_all[idx_from:idx_from+seq_length]
    batch_y = labels_all[idx_from:idx_from+seq_length]
    # use histogram to select a unique lable
    batch_y = batch_y[1,1]

    return batch_x, batch_y

init = tf.global_variables_initializer()
steps = 100

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x, batch_y = next_batch(i, seq_length, batch_size)
        
        sess.run(train, feed_dict={X:batch_x, y:batch_y})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc,feed_dict={X:batch_x, y:batch_y}))
            print('\n')