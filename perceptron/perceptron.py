# -*- coding: utf-8 -*- 

import tensorflow as tf
import numpy as np
import argparse
import sys

import dataset
from numpy.random import RandomState

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
    
def inference(input_tensor, nn_size):
    pass

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
    x = tf.placeholder(tf.float32, shape=(None,2), name="x-input")
    y = tf.placeholder(tf.float32, shape=(None,1),name="y-input")
    #dataset.draw_dataset(ds)

    #loss = tf.Variable(x, name="loss")

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #print sess.run(loss, feed_dict={x:input_np, y:label_np})

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_type', type=str, 
        help='dataset type: linear circle xor screw', default='linear')
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
        help='data noise.', default='3')
    parser.add_argument('--nn_size', type=int, nargs="+",
        help='full connect neural network.', default=[2,1])
    parser.add_argument('--random_seed', type=int,
        help='random seed.', default='0')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
