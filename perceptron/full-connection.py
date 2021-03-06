# -*- coding: utf-8 -*- 

import tensorflow as tf
import numpy as np
import argparse
import sys
import time

import dataset
from numpy.random import RandomState

#生成网格数据
def get_meshgrid(low, high, scale):
    x1 = np.arange(low, high, scale)
    x2 = np.arange(low, high, scale)
    grid_x1,grid_x2 = np.meshgrid(x1, x2)

    shape = grid_x1.shape

    x1_np = grid_x1.reshape((shape[0]*shape[1],1))
    x2_np = grid_x2.reshape((shape[0]*shape[1],1))
    x_np = np.hstack([x1_np, x2_np])
    return grid_x1, grid_x2, x_np

#由dataset生成输入张量
def get_input_from_dataset(ds):

    input_np = np.concatenate(ds)
    label_list = list()
    for i in range(len(ds)):
        label_list.append(np.full((ds[i].shape[0],1),i))
    #数组拼接
    label_np = np.concatenate(label_list)

    #水平合并数组
    all_np = np.hstack([input_np, label_np])

    #打乱顺序
    np.random.shuffle(all_np)

    #分割数据和标记
    return np.hsplit(all_np, [-1])
    
def get_full_connection_variable(depth, shape):
    #print depth,shape,"layer"+str(depth)

    #reuse 是否复用命名
    with tf.variable_scope("layer"+str(depth), reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights", shape, initializer = tf.truncated_normal_initializer(0, stddev=1))
        biases = tf.get_variable("biases", shape[1], initializer = tf.random_uniform_initializer(-1,1))
        return weights, biases

def inference(input_tensor, nn_shape):
    output_tensor = input_tensor

    #隐藏层使用relu激活
    for i,v in enumerate(nn_shape[:-2]):
        w, b = get_full_connection_variable(i, nn_shape[i:i+2])
        output_tensor = tf.nn.relu(tf.matmul(output_tensor, w) + b)

    #最后一层不使用激活函数
    i=i+1
    w, b = get_full_connection_variable(i, nn_shape[i:i+2])
    output_tensor = tf.matmul(output_tensor, w) + b

    return output_tensor

def calc_err(x, y, y_):
    return tf.reduce_sum(tf.where(tf.equal(y,y_), tf.zeros_like(y), tf.ones_like(y)))

def calc_loss(y_inf, y__softmax):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y__softmax, logits=y_inf))

def main(args):
    rdm = RandomState(args.random_seed)
    
    if args.dataset_type == 'linear':
        ds = dataset.generate_linear_dataset(rdm, args.class_num, args.data_num, args.linear_center_points, args.noise)
    elif args.dataset_type == 'circle':
        ds = dataset.generate_circle_dataset(rdm, args.class_num, args.data_num, args.circle_radius_limit, args.noise)
    elif args.dataset_type == 'xor':
        ds = dataset.generate_xor_dataset(rdm, args.class_num, args.data_num, args.xor_center_points, args.noise)
    elif args.dataset_type == 'screw':
        ds = dataset.generate_screw_dataset(rdm, args.class_num, args.data_num, args.screw_a_b, args.noise)
    else:
        pass

    input_np, label_np = get_input_from_dataset(ds)
    label_softmax_np = np.zeros((label_np.shape[0],2), dtype=float)
    #print label_np.shape[0]
    for i in range(label_np.shape[0]):
        j = label_np[i][0]
        label_softmax_np[i][int(j)] = 1

    x = tf.placeholder(tf.float32, shape=(None,2), name="x-input")
    y_ = tf.placeholder(tf.float32, shape=(None,1),name="y-input")
    y__softmax = tf.placeholder(tf.float32, shape=(None,2),name="y-softmax-input")

    #得到权值
    #wx, bx = get_full_connection_variable(0, (2,1))

    #计算输出
    y_inf = inference(x, args.nn_shape)
    y = tf.to_float(tf.expand_dims(tf.argmax(y_inf, axis = 1), 1)) #扩展维度，维度改为(None,1)

    #计算error
    err = calc_err(x,y,y_)

    #计算loss
    loss = calc_loss(y_inf, y__softmax)

    #计算等高线
    grid_x = tf.placeholder(tf.float32, shape=(None,2), name="grid-x-input")
    grid_x1, grid_x2, grid_input_np = get_meshgrid(args.mashgrid_range[0], args.mashgrid_range[1], args.mashgrid_range[2])
    grid_y_inf = inference(grid_x, args.nn_shape)
    grid_y = tf.to_float(tf.expand_dims(tf.argmax(grid_y_inf, axis = 1), 1))

    #开始训练
    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss, global_step=global_step)


    old_loss_c = -1
    old_err_c = -1

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #开启交互式
        dataset.draw_ion()
        for i in range(args.train_num):
            y_np, l_np, err_c, loss_c = sess.run([y, y_, err, loss], feed_dict={x:input_np, y_:label_np, y__softmax: label_softmax_np})
            print err_c, loss_c
            #print y_np.shape
            #print l_np.shape

            #wx_np, bx_np = sess.run([wx, bx])
            #print wx_np, bx_np

            if abs(old_err_c - err_c) != 0 or (i % 1000 == 0) or err_c == 0:
                dataset.draw_clear()
                #绘制等高线
                grid_y_np = sess.run(grid_y, feed_dict={grid_x:grid_input_np})
                print grid_y_np.shape
                grid_y_np_reshape = grid_y_np.reshape(grid_x1.shape)
                err_text = "fail: " + str(int(err_c)) + "/" + str(input_np.shape[0])
                loss_text = "loss: " + str(loss_c)
                text = err_text + "\n" + loss_text
                dataset.draw_contourf(grid_x1, grid_x2, grid_y_np_reshape, 1, text)

                #绘制散点图
                dataset.draw_dataset(ds, reverse=True)
                dataset.draw_pause(0.1)     
                      
            old_err_c = err_c
            old_loss_c = loss_c

            #训练
            if err_c == 0:
                break
            else:
                sess.run(train_step, feed_dict={x:input_np, y_:label_np, y__softmax: label_softmax_np})

        #关闭交互式
        dataset.draw_ioff()
        dataset.draw_show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_type', type=str, 
        help='dataset type: linear circle xor screw', default='circle')
    parser.add_argument('--class_num', type=int, 
        help='classification number.', default='2')
    parser.add_argument('--data_num', type=int, nargs="+",
        help='data num.', default=[50,50])
    parser.add_argument('--linear_center_points', type=int, nargs="+",
        help='center points coordinate. [x1,y1,x2,y2...]', default=[5,5,-5,-5])
    parser.add_argument('--circle_radius_limit', type=int, nargs="+",
        help='circle radius limit. [r1_min,r1_max,r2_min,r2_max...]', default=[0,3,5,7])
    parser.add_argument('--xor_center_points', type=int, nargs="+",
        help='center points coordinate. [x1,y1,x2,y2...]', default=[5,5,5,-5])
    parser.add_argument('--screw_a_b', type=int, nargs="+",
        help='screw a b. [a1,b1,a2,b2...]', default=[1,1,-1,-1])
    parser.add_argument('--noise', type=int,
        help='data noise.', default='5')
    parser.add_argument('--mashgrid_range', type=float, nargs="+",
        help='mashgrid range.', default=[-15.0,15.0, 0.01])
    parser.add_argument('--nn_shape', type=int, nargs="+",
        help='full connect neural network.', default=[2,5,5,2])
    parser.add_argument('--random_seed', type=int,
        help='random seed.', default='0')
    parser.add_argument('--learning_rate', type=float,
        help='learning rate.', default='0.001')
    parser.add_argument('--train_num', type=int,
        help='train num.', default='100000')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
