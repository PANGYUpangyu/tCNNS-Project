import tensorflow as tf
from batcher import *
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weight_variable(shape):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    return tf.compat.v1.Variable(initial)

def bias_variable(shape):
    initial = tf.compat.v1.constant(0.1, shape=shape)
    return tf.compat.v1.Variable(initial)

def conv1d(x, W):
    return tf.compat.v1.nn.conv1d(x, W, 1, padding='SAME')

def max_pool_1d(x, kernel_shape, strides, padding='SAME'):
    return tf.compat.v1.nn.pool(x, kernel_shape, 'MAX', padding=padding, strides=strides)

def R2(real, predicted):
    label_mean = tf.compat.v1.reduce_mean(real, axis=0)
    total_sum_of_square = tf.compat.v1.reduce_sum(tf.compat.v1.square(real - label_mean), axis=0)
    residual_sum_of_square = tf.compat.v1.reduce_sum(tf.compat.v1.square(real - predicted), axis=0)
    r2 = 1 - residual_sum_of_square / total_sum_of_square
    return r2

def Pearson(a, b):
    real = tf.compat.v1.squeeze(a)
    pred = tf.compat.v1.squeeze(b)
    real_new = real - tf.compat.v1.reduce_mean(real)
    pred_new = pred - tf.compat.v1.reduce_mean(real)
    up = tf.compat.v1.reduce_mean(tf.compat.v1.multiply(real_new, pred_new))
    real_var = tf.compat.v1.reduce_mean(tf.compat.v1.multiply(real_new, real_new))
    pred_var = tf.compat.v1.reduce_mean(tf.compat.v1.multiply(pred_new, pred_new))
    down = tf.compat.v1.multiply(tf.compat.v1.sqrt(real_var), tf.compat.v1.sqrt(pred_var))
    return tf.compat.v1.divide(up,down)

tf.compat.v1.disable_eager_execution()
drug = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 188, 31])
cell = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 425])
scores = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, 1])
keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32)

drug_conv1_out = 40
drug_conv1_pool = 3
drug_conv1_w = weight_variable([7, 31, drug_conv1_out])
drug_conv1_b = bias_variable([drug_conv1_out])
drug_conv1_h = tf.compat.v1.nn.relu(conv1d(drug, drug_conv1_w) + drug_conv1_b)
drug_conv1_p = max_pool_1d(drug_conv1_h, [drug_conv1_pool], [drug_conv1_pool])

drug_conv2_out = 80
drug_conv2_pool = 3
drug_conv2_w = weight_variable([7, drug_conv1_out, drug_conv2_out])
drug_conv2_b = bias_variable([drug_conv2_out])
drug_conv2_h = tf.compat.v1.nn.relu(conv1d(drug_conv1_p, drug_conv2_w) + drug_conv2_b)
drug_conv2_p = max_pool_1d(drug_conv2_h, [drug_conv2_pool], [drug_conv2_pool])

drug_conv3_out = 60
drug_conv3_pool = 3
drug_conv3_w = weight_variable([7, drug_conv2_out, drug_conv3_out])
drug_conv3_b = bias_variable([drug_conv3_out])
drug_conv3_h = tf.compat.v1.nn.relu(conv1d(drug_conv2_p, drug_conv3_w) + drug_conv3_b)
drug_conv3_p = max_pool_1d(drug_conv3_h, [drug_conv3_pool], [drug_conv3_pool])

cell_conv1_out = 40
cell_conv1_pool = 3
cell_tensor = tf.compat.v1.expand_dims(cell, 2)
cell_conv1_w = weight_variable([7, 1, cell_conv1_out])
cell_conv1_b = weight_variable([cell_conv1_out])
cell_conv1_h = tf.compat.v1.nn.relu(conv1d(cell_tensor, cell_conv1_w) + cell_conv1_b)
cell_conv1_p = max_pool_1d(cell_conv1_h, [cell_conv1_pool], [cell_conv1_pool])

cell_conv2_out = 80
cell_conv2_pool = 3
cell_conv2_w = weight_variable([7, cell_conv1_out, cell_conv2_out])
cell_conv2_b = bias_variable([cell_conv2_out])
cell_conv2_h = tf.compat.v1.nn.relu(conv1d(cell_conv1_p, cell_conv2_w) + cell_conv2_b)
cell_conv2_p = max_pool_1d(cell_conv2_h, [cell_conv2_pool], [cell_conv2_pool])

cell_conv3_out = 60
cell_conv3_pool = 3
cell_conv3_w = weight_variable([7, cell_conv2_out, cell_conv3_out])
cell_conv3_b = bias_variable([cell_conv3_out])
cell_conv3_h = tf.compat.v1.nn.relu(conv1d(cell_conv2_p, cell_conv3_w) + cell_conv3_b)
cell_conv3_p = max_pool_1d(cell_conv3_h, [cell_conv3_pool], [cell_conv3_pool])

conv_merge = tf.compat.v1.concat([drug_conv3_p, cell_conv3_p], 1)
shape = conv_merge.get_shape().as_list()
conv_flat = tf.compat.v1.reshape(conv_merge, [-1, shape[1] * shape[2]])

fc1_w = weight_variable([shape[1] * shape[2], 1024])
fc1_b = bias_variable([1024])
fc1_h = tf.compat.v1.nn.relu(tf.compat.v1.matmul(conv_flat, fc1_w) + fc1_b)
fc1_drop = tf.compat.v1.nn.dropout(fc1_h, rate = 1 - keep_prob)

fc2_w = weight_variable([1024, 1024])
fc2_b = bias_variable([1024])
fc2_h = tf.compat.v1.nn.relu(tf.compat.v1.matmul(fc1_drop, fc2_w) + fc2_b)
fc2_drop = tf.compat.v1.nn.dropout(fc2_h, rate = 1 - keep_prob)

fc3_w = weight_variable([1024, 1])
fc3_b = weight_variable([1])

y_conv = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(fc2_drop, fc3_w) + fc3_b)
loss = tf.compat.v1.losses.mean_squared_error(scores, y_conv)
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)

r_square = R2(scores, y_conv)
pearson = Pearson(scores, y_conv)
rmsr = tf.compat.v1.sqrt(loss)

for i in range(0,2):
    train, valid, test, test1 = load_data(100, ['IC50'])
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.trainable_variables())
    output_file = open("result_all", "a")
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        test_values, test_drugs, test_cells = test.whole_batch()
        valid_values, valid_drugs, valid_cells = valid.whole_batch()
        test1_values, test1_drugs, test1_cells = test1.whole_batch()
        epoch = 0
        min_loss = 100
        count = 0
        while count < 10:
            train.reset()
            step = 0
            while(train.available()):
                real_values, drug_smiles, cell_muts = train.mini_batch()
                train_step.run(feed_dict={drug:drug_smiles, cell:cell_muts, scores:real_values, keep_prob:0.5})
                step += 1
            valid_loss, valid_r2, valid_p, valid_rmsr = sess.run([loss, r_square, pearson, rmsr], feed_dict={drug:valid_drugs, cell:valid_cells, scores:valid_values, keep_prob:1})
            print("epoch: %d, loss: %g r2: %g pearson: %g rmsr: %g" % (epoch, valid_loss, valid_r2, valid_p, valid_rmsr))
            epoch += 1
            if valid_loss < min_loss:
                test_loss, test_r2, test_p, test_rmsr = sess.run([loss, r_square, pearson, rmsr], feed_dict={drug:test_drugs, cell:test_cells, scores:test_values, keep_prob:1})
                print("find best, loss: %g r2: %g pearson: %g rmsr: %g ******" % (test_loss, test_r2, test_p, test_rmsr))
                y = sess.run([y_conv], feed_dict={drug: test1_drugs, cell: test1_cells, keep_prob: 1})
                for x in y:
                    for z in x:
                        z = -10 * math.log(1 / z - 1)
                        print(z)
                os.system("rm model_all")
                saver.save(sess, "./model_all/result.ckpt")
                print("saved!")
                min_loss = valid_loss
                count = 0
            else:
                count = count + 1


        if test_r2 > -2:
            output_file.write("%g,%g,%g,%g\n" % (test_loss, test_r2, test_p, test_rmsr))
            print("Saved!!!!!")
        output_file.close()