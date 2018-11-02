#!/usr/bin/env python
# coding: utf-8

# In[1]:




import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

# 因为slim.nets包在 tensorflow 1.3 中有一些问题，所以这里为了方便
# 我们将slim.nets.inception_v3中的代码拷贝到了同一个文件夹下。
import inception_v3

# 处理好之后的数据文件。
INPUT_DATA = 'flower_processed_data.npy'
# 保存训练好的模型的路径。这里我们可以将使用新数据训练得到的完整模型保存
# 下来，如果计算资源充足，我们还可以在训练完最后的全联接层之后再训练所有
# 网络层，这样可以使得新模型更加贴近新数据。
TRAIN_FILE = 'train_dir/model'
# 谷歌提供的训练好的模型文件地址。
CKPT_FILE = 'inception_v3.ckpt'

# 定义训练中使用的参数。
LEARNING_RATE = 0.01
STEPS = 5000
BATCH = 128
N_CLASSES = 5

# 不需要从谷歌训练好的模型中加载的参数。这里就是最后的全联接层，因为在
# 新的问题中我们要重新训练这一层中的参数。这里给出的是参数的前缀。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
# 这里给出的是参数的前缀。
TRAINABLE_SCOPES='InceptionV3/Logits'

# 获取所有需要从谷歌训练好的模型中加载的参数。
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中
    # 移除。
    for var in slim.get_model_variables():
        print var.op.name
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

# 获取所有需要训练的变量列表。
def get_trainable_variables():    
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有的参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train
    
def main():
    # 加载预处理好的数据。
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    # 定义inception-v3的输入，images为输入图片，labels为每一张图片
    # 对应的标签。
    images = tf.placeholder(
        tf.float32, [None, 299, 299, 3], 
        name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')
    
    # 定义inception-v3模型。因为谷歌给出的只有模型参数取值，所以这里
    # 需要在这个代码中定义inception-v3的模型结构。因为模型
    # 中使用到了dropout，所以需要定一个训练时使用的模型，一个测试时
    # 使用的模型。
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        train_logits, _ = inception_v3.inception_v3(
            images, num_classes=N_CLASSES, is_training=True)
        # 定义测试使用的模型时需要将reuse设置为True。
        test_logits, _ = inception_v3.inception_v3(
            images, num_classes=N_CLASSES, is_training=False, reuse=True)
    
    
    trainable_variables = get_trainable_variables()
    print trainable_variables
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=train_logits, 
        labels=tf.one_hot(labels, N_CLASSES))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        cross_entropy_mean,
        var_list=trainable_variables)
    
    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(test_logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 
    loader = tf.train.Saver(get_tuned_variables())
    saver = tf.train.Saver()
    with tf.variable_scope("InceptionV3", reuse = True):
        check1 = tf.get_variable("Conv2d_1a_3x3/weights")
        check2 = tf.get_variable("Logits/Conv2d_1c_1x1/weights")
        
    with tf.Session() as sess:
        # 初始化没有加载进来的变量。
        init = tf.global_variables_initializer()
        sess.run(init)
        print sess.run(check1)
        print sess.run(check2)
        
        # 加载谷歌已经训练好的模型。
        print('Loading tuned variables from %s' % CKPT_FILE)
        loader.restore(sess, CKPT_FILE)
            
        start = 0
        end = BATCH
        for i in range(STEPS):
            print sess.run(check1)
            print sess.run(check2)
            
            _, loss = sess.run([train_step, cross_entropy_mean], feed_dict={
                images: training_images[start:end], 
                labels: training_labels[start:end]})

            if i % 100 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels})
                print('Step %d: Training loss is %.1f%% Validation accuracy = %.1f%%' % (
                    i, loss * 100.0, validation_accuracy * 100.0))
            
            start = end
            if start == n_training_example:
                start = 0
            
            end = start + BATCH
            if end > n_training_example: 
                end = n_training_example
            
        # 在最后的测试数据上测试正确率。
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: test_images, labels: test_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()


# In[ ]:




