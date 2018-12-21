import matplotlib.pyplot as plt;  
import tensorflow as tf;  
import numpy as np;
 
with tf.Session() as sess:
	image_raw_data_jpg = tf.gfile.FastGFile('gyy.jpg', 'r').read()
	image_data = tf.image.decode_jpeg(image_raw_data_jpg)
	print image_data.eval().shape
	image_data = sess.run(tf.image.rgb_to_grayscale(image_data))
	print image_data.shape
	plt.imshow(image_data[:,:,0],cmap ='gray')
	plt.show()

