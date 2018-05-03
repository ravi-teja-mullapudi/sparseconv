from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import sys

def run_sub_graph(out_dict, in_dict, num_runs):
    res = {}
    best_time = float('inf')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(0, num_runs):
            start_time = time.time() * 1000
            res = sess.run(out_dict, feed_dict = in_dict)
            end_time = time.time() * 1000
            best_time = min(best_time, end_time - start_time)
    return (res, best_time)

def benchmark_conv2d(batch_size,
                     in_h,
                     in_w,
                     c_i,
                     k_h,
                     k_w,
                     c_o,
                     s_h,
                     s_w,
                     num_runs = 5,
                     device = '/gpu:0'):
    with tf.variable_scope('conv2d'):
        with tf.device(device):
           input = tf.placeholder(tf.float32, shape = [batch_size, in_h, in_w, c_i])
           w = tf.get_variable("boxes",
                               initializer = np.random.rand(k_h, k_w, c_i, c_o).astype('f'))
           b = tf.get_variable("biases", initializer = np.random.rand(c_o).astype('f'))

           w_out = tf.nn.conv2d(input, w, [1, s_h, s_w, 1], padding="VALID")
           out = tf.nn.bias_add(w_out, b)

    # Create a random input
    rand_in = np.random.rand(batch_size, in_h, in_w, c_i)
    # Run the sub graph multiple times
    in_dict = { input : rand_in }
    out_dict = { "out" : out }
    _, best_time = run_sub_graph(out_dict, in_dict, num_runs)
    return best_time

def do_one_layer(layer, c_i, c_o, in_h, in_w):
    batch_size = 1
    #in_w = 256
    #in_h = 256
    #c_i = 16
    k_h = 3
    k_w = 3
    #c_o = 64
    s_h = 1
    s_w = 1
    out_w = in_w/s_w
    out_h = in_h/s_h
    t = benchmark_conv2d(batch_size,
                           in_h,
                           in_w,
                           c_i,
                           k_h,
                           k_w,
                           c_o,
                           s_h,
                           s_w,
                           num_runs = 10,
                           device='/cpu:0')
    gops = (batch_size * out_w * out_h * c_i * c_o * k_w * k_h)/1e09
    gflops = float(gops)/(t/1e03)
    print("Layer {}, {}, {}".format(layer, t, gflops))

def main():
    hw_arr =   [226, 226, 114, 114,  58,  58,  58,  30,  30,  30,  16,  16,  16]
    cin_arr =  [  3,  64,  64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
    cout_arr = [ 64,  64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    layer = int(sys.argv[1])
    cin = int(sys.argv[2])
    cout = int(sys.argv[3])
    h = int(sys.argv[4])
    w = h

    print("Layer: {}, C_in: {}, C_out: {}, h x w: {} x {}".format(layer, cin, cout, h, w))

    do_one_layer(layer, cin, cout, h, w)

    #for layer in range(len(hw_arr)):
    #    cin = cin_arr[layer]
    #    cout = cout_arr[layer]
    #    h = hw_arr[layer]
    #    w = hw_arr[layer]
    #    do_one_layer(layer, cin, cout, h, w)

if __name__ == '__main__':
    main()
