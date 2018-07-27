#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:13:20 2018

@author: simon
"""
import tensorflow as tf
import sys, json, base64
import skvideo.io
import scipy.misc
import timeit
import numpy as np
import scipy.misc as misc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

def main():
    
    image_shape = (600, 800)
    frame = 2
    
    file = sys.argv[-1]
    myname = __file__
    if file == myname:
        print ("Error loading video")
        exit()
    
    #video = skvideo.io.vread(file)
    img_files = []
    for i in range(1):
        img_files.append(file)
    #img = misc.imread(img_files)

    with tf.gfile.GFile('./frozen_graph.pb', 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    G = tf.Graph()
    
    with tf.Session(graph=G) as sess:
        logits, = tf.import_graph_def(graph_def_optimized, return_elements=['network/output/ArgMax:0'])
        image_input = G.get_tensor_by_name('import/network/input/Placeholder:0')
        is_training = G.get_tensor_by_name('import/network/input/Placeholder_2:0')

       # keep_prob = G.get_tensor_by_name('import/keep_prob:0')

        sess.run(tf.global_variables_initializer())

        start_time = timeit.default_timer()
       
        #test_input = np.asarray(img, dtype='float')
        #for i in range(5):
        #    test_input.append(img)
        
        #test_input = convert_to_tensor(img_files, dtype= dtypes.string)
        test_input = np.ndarray(shape=(len(img_files), image_shape[0], image_shape[1], 3), dtype=np.float32)    
        print("test_input size is {}".format(test_input.shape))
        i = 0
        for _file in img_files:
            _img = misc.imread(_file)
            test_input[i] = np.asarray(_img, dtype='float32')
            i += 1
        result_img = sess.run(
            [logits],
            {image_input: test_input, is_training: False})

        print("result is {}".format(result_img[0].shape))
        road_seg = (result_img[0][0] == 1).reshape(image_shape[0], image_shape[1], 1)
        car_seg = (result_img[0][0] == 2).reshape(image_shape[0], image_shape[1], 1)

        road_mask = np.dot(road_seg, np.array([[0, 255, 0, 127]]))
        car_mask = np.dot(car_seg, np.array([[0, 255, 0, 127]]))

        road_mask = scipy.misc.toimage(road_mask, mode="RGBA")
        car_mask = scipy.misc.toimage(car_mask, mode="RGBA")
 
        road_im = misc.imread(img_files[0])
        road_im = scipy.misc.toimage(road_im)
        road_im.paste(road_mask, box=None, mask=road_mask)
        #road_im = scipy.misc.imresize(road_im, original_image_shape)
 
        car_im = misc.imread(img_files[0])
        car_im = scipy.misc.toimage(car_im)
        car_im.paste(car_mask, box=None, mask=car_mask)
        #car_im = scipy.misc.imresize(car_im, original_image_shape)
 
        road_im_path = "./road_seg.png"
        car_im_path = "./car_seg.png"
 
        scipy.misc.imsave(road_im_path, np.array(road_im))
        scipy.misc.imsave(car_im_path, np.array(car_im))
        #print("Save the result to result.npy")
        #np.save('./result_img.npy', result_img)
 
        """
        for rgb_frame in video:
            original_image_shape = (rgb_frame.shape[0], rgb_frame.shape[1])
            rgb_frame_scaled = scipy.misc.imresize(rgb_frame, image_shape)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_input: [rgb_frame_scaled]})
        
            elapsed = timeit.default_timer() - start_time
            print("inferencing time is {}".format(elapsed))
            
            print("im_softmax shape is {}".format(im_softmax[0][0, :, :, 0].shape))

            #road_result = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            #car_result = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
            road_result = im_softmax[0][0, :, :, 1]
            car_result = im_softmax[0][0, :, :, 2]
            
            road_seg = (road_result > 0.5).reshape(image_shape[0], image_shape[1], 1)
            car_seg = (car_result > 0.5).reshape(image_shape[0], image_shape[1], 1)
            
            road_mask = np.dot(road_seg, np.array([[0, 255, 0, 127]]))
            car_mask = np.dot(car_seg, np.array([[0, 255, 0, 127]]))
            
            road_mask = scipy.misc.toimage(road_mask, mode="RGBA")
            car_mask = scipy.misc.toimage(car_mask, mode="RGBA")
            
            road_im = scipy.misc.toimage(rgb_frame_scaled)
            road_im.paste(road_mask, box=None, mask=road_mask)
            road_im = scipy.misc.imresize(road_im, original_image_shape)
            
            car_im = scipy.misc.toimage(rgb_frame_scaled)
            car_im.paste(car_mask, box=None, mask=car_mask)
            car_im = scipy.misc.imresize(car_im, original_image_shape)
            
            road_im_path = "road_" + str(frame) + ".png"
            car_im_path = "car_" + str(frame) + ".png"
            
            scipy.misc.imsave(road_im_path, np.array(road_im))
            scipy.misc.imsave(car_im_path, np.array(car_im))
        """


if __name__ == '__main__':
    main()
    
                                      
                                      
