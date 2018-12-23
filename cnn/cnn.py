# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import sys
 
def main(args):
    image_raw_data = tf.gfile.FastGFile(args.image, 'r').read()
    image_data_rgb = tf.image.decode_jpeg(image_raw_data)
    #image_data_rgb = tf.image.convert_image_dtype(image_data_rgb, dtype=tf.float32)
    #image_data_rgb = image_data_rgb * 255

    #灰度图
    image_data_gray = tf.image.rgb_to_grayscale(image_data_rgb)

    #conv 2d
    #conv_filter = tf.Variable(tf.truncated_normal([2,2,3,1],stddev=0.1))
    conv_filter1 = tf.constant(
        [[[[ -1.0],[ -1.0], [ -1.0]],
      [[1.0], [ 1.0], [1.0]]],
     [[[ -1.0],[ -1.0], [ -1.0]],
      [[1.0], [1.0],[ 1.0 ]]]])

    conv_filter2 = tf.constant(
        [[[[ -1.0],[ -1.0], [ -1.0]],
      [[-1.0], [ -1.0], [-1.0]]],
     [[[ 1.0],[ 1.0], [ 1.0]],
      [[1.0], [1.0],[ 1.0 ]]]])

    #print conv_filter1.shape

    image_data_conv1 = tf.nn.conv2d(tf.expand_dims(tf.to_float(image_data_rgb), 0),  #扩展维度到4
                                         conv_filter1, strides=[1,1,1,1], padding='SAME')

    image_data_conv2 = tf.nn.conv2d(tf.expand_dims(tf.to_float(image_data_rgb), 0),  #扩展维度到4
                                         conv_filter2, strides=[1,1,1,1], padding='SAME')

    #max pool
    image_data_max_pool = tf.nn.max_pool(tf.expand_dims(tf.to_float(image_data_rgb), 0),  #扩展维度到4
                                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #avg pool
    image_data_avg_pool = tf.nn.avg_pool(tf.expand_dims(tf.to_float(image_data_rgb), 0),  #扩展维度到4
                                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        if args.action == "gray":
            data_rgb_np, data_gray_np = sess.run([image_data_rgb, image_data_gray])
            print data_rgb_np.shape
            print data_gray_np.shape

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(data_rgb_np)

            plt.subplot(1,2,2)
            plt.imshow(data_gray_np[:,:,0], cmap ='gray')

            plt.show()
        elif args.action == "conv":
            data_rgb_np, conv_filter1_np, conv_filter2_np, image_data_conv1, image_data_conv2 = sess.run(
                [image_data_rgb, conv_filter1, conv_filter2, image_data_conv1, image_data_conv2])
            print data_rgb_np.shape, image_data_conv1.shape, image_data_conv2.shape
            #print conv_filter_np1

            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(data_rgb_np)

            plt.subplot(1,3,2)
            plt.imshow(image_data_conv1[0,:,:,0], cmap ='gray')

            plt.subplot(1,3,3)
            plt.imshow(image_data_conv2[0,:,:,0], cmap ='gray')

            plt.show()
            
        elif args.action == "pool":
            data_rgb_np, image_data_max_pool_np, image_data_avg_pool_np = sess.run([image_data_rgb, image_data_max_pool, image_data_avg_pool])

            print data_rgb_np.shape,image_data_max_pool_np.shape,image_data_avg_pool_np.shape
            #print image_data_max_pool_np

            plt.figure()

            plt.subplot(1,3,1)
            plt.imshow(data_rgb_np)
            #np.savetxt("data_rgb_np[0].txt",data_rgb_np[:,:,0])
            #np.savetxt("data_rgb_np[1].txt",data_rgb_np[:,:,1])
            #np.savetxt("data_rgb_np[2].txt",data_rgb_np[:,:,2])
            #print data_rgb_np

            plt.subplot(1,3,2)
            plt.imshow(image_data_max_pool_np[0,:,:,:].astype(np.uint8))

            plt.subplot(1,3,3)
            plt.imshow(image_data_avg_pool_np[0,:,:,:].astype(np.uint8))

            plt.show()
        else:
            pass

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, 
        help='input image path', default='./gyy.jpg')

    parser.add_argument('--action', type=str, 
        help='action: gray conv pool', default='conv')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
