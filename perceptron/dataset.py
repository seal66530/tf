# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from numpy.random import RandomState

#生成线性可分的测试集
def generate_linear_dataset(rdm, class_num, data_num, center_points, noise):
    print("class_num: %s" % class_num)
    print("data_num: %s" % data_num)
    print("center_points: %s" % center_points)
    print("noise: %s" % noise)

    all_nps = []

    for i in range(class_num):
        mean = [center_points[i*2], center_points[i*2+1] ]
        cov = [[1+noise, 0], [0, 1+noise]]
        ds = rdm.multivariate_normal(mean, cov, data_num[i])
        all_nps.append(ds)

    return all_nps

#生成圆环形状的测试集
def generate_circle_dataset(rdm, class_num, data_num, radius_limit, noise):
    print("class_num: %s" % class_num)
    print("data_num: %s" % data_num)
    print("radius_limit: %s" % radius_limit)
    print("noise: %s" % noise)

    all_nps = []

    for i in range(class_num):
        ds = np.zeros((data_num[i],2), dtype=float)
        radius = rdm.uniform(radius_limit[i*2], radius_limit[i*2+1], data_num[i])
        #radius = radius + rdm.uniform(1,10) * noise
        angle = rdm.uniform(0, 2*np.pi, data_num[i])
        ds[:,0] = np.cos(angle) * radius
        ds[:,1] = np.sin(angle) * radius
        all_nps.append(ds)

    return all_nps

#生成异或形状的测试集
def generate_xor_dataset(rdm, class_num, data_num, center_points, noise):
    print("class_num: %s" % class_num)
    print("data_num: %s" % data_num)
    print("center_points: %s" % center_points)
    print("noise: %s" % noise)

    all_nps = []

    for i in range(class_num):
        mean1 = [center_points[i*2], center_points[i*2+1] ]
        cov1 = [[1+noise, 0], [0, 1+noise]]
        ds1 = rdm.multivariate_normal(mean1, cov1, data_num[i]/2)

        mean2 = [-center_points[i*2], -center_points[i*2+1] ]
        cov2 = [[1+noise, 0], [0, 1+noise]]
        ds2 = rdm.multivariate_normal(mean2, cov2, data_num[i]/2)

        ds=np.concatenate([ds1,ds2])
        all_nps.append(ds)

    return all_nps

#生成螺线形状的测试集
def generate_screw_dataset(rdm, class_num, data_num, screw_a_b, noise):
    print("class_num: %s" % class_num)
    print("data_num: %s" % data_num)
    print("screw_a_b: %s" % screw_a_b)
    print("noise: %s" % noise)
    
    all_nps = []

    for i in range(class_num):
        ds = np.zeros((data_num[i],2), dtype=float)
        angle = rdm.uniform(0, 4*np.pi, data_num[i] )
        radius = screw_a_b[i*2] + screw_a_b[i*2+1] * angle
        ds[:,0] = np.cos(angle) * radius
        ds[:,1] = np.sin(angle) * radius
        all_nps.append(ds)

    return all_nps

#绘图
def draw_dataset(dataset):
    colors = [ 'r', 'b', 'g' ]
    i = 0

    for np in dataset:
        X=np[:,0]
        Y=np[:,1]
        plt.scatter(X,Y, marker='o', c=colors[i])
        i=i+1

    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.axis('equal')
    plt.grid()

def draw_contourf(x, y, f, text=""):
    plt.contourf(x, y, f, 0, cmap = plt.cm.cool)

    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.axis('equal')
    plt.grid()
    plt.text(15,16,text, fontsize=12)
    #plt.figure(figsize=(8,8))

def draw_contour(x, y, f,text=""):
    plt.contour(x, y, f, 0, cmap = plt.cm.cool)

    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.axis('equal')
    plt.grid()
    plt.text(15,16,text, fontsize=12)
    #plt.figure(figsize=(10,10))

def draw_pause(sec):
    plt.pause(sec)

def draw_show():
    plt.show()

def draw_ion():
    plt.ion()

def draw_ioff():
    plt.ioff()

def draw_clear():
    plt.cla()
    plt.clf()

def main(args):
    rdm = RandomState(args.random_seed)

    if len(args.data_num) != args.class_num:
        print("data_num is invalid.")
        return

    if args.dataset_type == 'linear':
        ds = generate_linear_dataset(rdm, args.class_num, args.data_num, args.linear_center_points, args.noise)
    elif args.dataset_type == 'circle':
        ds = generate_circle_dataset(rdm, args.class_num, args.data_num, args.circle_radius_limit, args.noise)
    elif args.dataset_type == 'xor':
        ds = generate_xor_dataset(rdm, args.class_num, args.data_num, args.xor_center_points, args.noise)
    elif args.dataset_type == 'screw':
        ds = generate_screw_dataset(rdm, args.class_num, args.data_num, args.screw_a_b, args.noise)
    else:
        pass

    draw_dataset(ds)
    draw_show()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_type', type=str, 
        help='dataset type: linear circle xor screw', default='screw')
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
    parser.add_argument('--random_seed', type=int,
        help='random seed.', default='0')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
