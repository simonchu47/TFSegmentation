from models.basic.basic_model import BasicModel
from models.encoders.shufflenet import ShuffleNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d
from utils.misc import get_vars_underscope

import tensorflow as tf
import pdb

class FCN8sShuffleNet(BasicModel):
    """
    FCN8s with ShuffleNet as an encoder Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
        # init network layers

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

    def init_network(self):
        """
        Building the Network here
        :return:
        """

        # Init ShuffleNet as an encoder
        self.encoder = ShuffleNet(x_input=self.x_pl, num_classes=self.params.num_classes,
                                  pretrained_path=self.args.pretrained_path, train_flag=self.is_training,
                                  batchnorm_enabled=self.args.batchnorm_enabled, num_groups=self.args.num_groups,
                                  weight_decay=self.args.weight_decay, bias=self.args.bias)

        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('upscore_2s'):
            #self.upscore2 = conv2d_transpose('upscore2', x=self.encoder.score_fr,
            #                                 output_shape=self.encoder.feed1.shape.as_list()[0:3] + [
            #                                     self.params.num_classes], batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
            #                                 kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd, bias=self.args.bias)
            upscore2_shape = self.encoder.feed1.shape.as_list()[0:3] + [
                self.params.num_classes]
            upscore2_shape[0] = tf.shape(self.encoder.feed1)[0]
            self.upscore2 = conv2d_transpose('upscore2', x=self.encoder.score_fr,
                                             output_shape=upscore2_shape, batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd, bias=self.args.bias)
                                             
#            currvars= get_vars_underscope(tf.get_variable_scope().name, 'upscore2')
#            for v in currvars:
#                tf.add_to_collection('decoding_trainable_vars', v)

            self.score_feed1 = conv2d('score_feed1', x=self.encoder.feed1, batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias= self.args.bias,
                                      l2_strength=self.encoder.wd)
#            currvars= get_vars_underscope(tf.get_variable_scope().name, 'score_feed1')
#            for v in currvars:
#                tf.add_to_collection('decoding_trainable_vars', v)


            self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        with tf.name_scope('upscore_4s'):
            #self.upscore4 = conv2d_transpose('upscore4', x=self.fuse_feed1,
            #                                 output_shape=self.encoder.feed2.shape.as_list()[0:3] + [
            #                                     self.params.num_classes], batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
            #                                 kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd, bias=self.args.bias,
            #                                 )
            upscore4_shape = self.encoder.feed2.shape.as_list()[0:3] + [
                self.params.num_classes]
            upscore4_shape[0] = tf.shape(self.encoder.feed2)[0]
            self.upscore4 = conv2d_transpose('upscore4', x=self.fuse_feed1,
                                             output_shape=upscore4_shape, batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd, bias=self.args.bias)

#            currvars= get_vars_underscope(tf.get_variable_scope().name, 'upscore4')
#            for v in currvars:
#                tf.add_to_collection('decoding_trainable_vars', v)

            self.score_feed2 = conv2d('score_feed2', x=self.encoder.feed2, batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias=self.args.bias,
                                      l2_strength=self.encoder.wd)
#            currvars= get_vars_underscope(tf.get_variable_scope().name, 'score_feed2')
#            for v in currvars:
#                tf.add_to_collection('decoding_trainable_vars', v)

            self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        with tf.name_scope('upscore_8s'):
            #self.upscore8 = conv2d_transpose('upscore8', x=self.fuse_feed2,
            #                                 output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes], is_training=self.is_training,
            #                                 kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd, bias=self.args.bias,
            #                                 )
            upscore8_shape = self.x_pl.shape.as_list()[0:3] + [self.params.num_classes]
            upscore8_shape[0] = tf.shape(self.x_pl)[0]
            self.upscore8 = conv2d_transpose('upscore8', x=self.fuse_feed2,
                                             output_shape=upscore8_shape, is_training=self.is_training,
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd, bias=self.args.bias)

#            currvars= get_vars_underscope(tf.get_variable_scope().name, 'upscore8')
#            for v in currvars:
#                tf.add_to_collection('decoding_trainable_vars', v)

        self.logits = self.upscore8
