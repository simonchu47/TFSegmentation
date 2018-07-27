#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:13:20 2018

@author: simon
"""
import tensorflow as tf
import sys
#import skvideo.io
#import scipy.misc
#import timeit
#import numpy as np

def main():
    
    #image_shape = (192, 256)
    #frame = 2
    
    file = sys.argv[-1]
    myname = __file__
    if file == myname:
        print ("Error loading forzen graph")
        exit()
    
    #video = skvideo.io.vread(file)
    
    with tf.gfile.GFile(file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    G = tf.Graph()
    
    with tf.Session(graph=G) as sess:
        g_in = tf.import_graph_def(graph_def)
        #image_input = G.get_tensor_by_name('import/image_input:0')
        #keep_prob = G.get_tensor_by_name('import/keep_prob:0')

        sess.run(tf.global_variables_initializer())

        #start_time = timeit.default_timer()
    LOGDIR = "./log_visualized"
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)

if __name__ == '__main__':
    main()
    
                                      
                                      
