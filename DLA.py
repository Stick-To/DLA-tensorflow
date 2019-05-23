from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


class DLA:
    def __init__(self, config, input_shape, num_classes, weight_decay, data_format):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        assert data_format in ['channels_last', 'channels_first']
        self.data_format = data_format
        self.config = config
        assert len(config['block_list']) == 6
        assert len(config['filters_list']) == 6
        self.is_bottleneck = config['is_bottleneck']
        self.is_groupconv = config['is_groupconv']
        self.block_list = config['block_list']
        self.filters_list = config['filters_list']

        self.global_step = tf.train.get_or_create_global_step()
        self.is_training = True

        self._define_inputs()
        self._build_graph()
        self._init_session()

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.input_shape)
        self.images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):

        conv = self._conv_bn_activation(
            bottom=self.images,
            filters=self.filters_list[0],
            kernel_size=7,
            strides=1,
        )
        with tf.variable_scope('stage1'):
            for i in range(self.block_list[0]):
                conv = self._conv_bn_activation(
                    bottom=conv,
                    filters=self.filters_list[0],
                    kernel_size=3,
                    strides=1,
                )
        with tf.variable_scope('stage2'):
            for i in range(self.block_list[1]-1):
                conv = self._conv_bn_activation(
                    bottom=conv,
                    filters=self.filters_list[1],
                    kernel_size=3,
                    strides=1,
                )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=self.filters_list[1],
                kernel_size=3,
                strides=2,
            )
        if self.is_bottleneck:
            stack_basic_fn = self._residual_bottleneck
        else:
            stack_basic_fn = self._basic_block
        with tf.variable_scope('stage3'):
            dla_stage3 = self._dla_generator(conv, self.filters_list[2], self.block_list[2]-1, stack_basic_fn)
            dla_stage3 = self._max_pooling(dla_stage3, 2, 2)
        with tf.variable_scope('stage4'):
            dla_stage4 = self._dla_generator(dla_stage3, self.filters_list[3], self.block_list[3]-1, stack_basic_fn)
            residual = self._conv_bn_activation(dla_stage3, self.filters_list[3], 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage4 = self._max_pooling(dla_stage4, 2, 2)
            dla_stage4 = dla_stage4 + residual
        with tf.variable_scope('stage5'):
            dla_stage5 = self._dla_generator(dla_stage4, self.filters_list[4], self.block_list[4]-1, stack_basic_fn)
            residual = self._conv_bn_activation(dla_stage4, self.filters_list[4], 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage5 = self._max_pooling(dla_stage5, 2, 2)
            dla_stage5 = dla_stage5 + residual
        with tf.variable_scope('stage6'):
            dla_stage6 = self._dla_generator(dla_stage5, self.filters_list[5], self.block_list[5]-1, stack_basic_fn)
            residual = self._conv_bn_activation(dla_stage5, self.filters_list[5], 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage6 = self._max_pooling(dla_stage6, 2, 2)
            dla_stage6 = dla_stage6 + residual
        with tf.variable_scope('final_dense'):
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(dla_stage6, axis=axes, keepdims=False, name='global_pool')
            final_dense = tf.layers.dense(global_pool, self.num_classes, name='final_dense')
        with tf.variable_scope('optimizer'):
            self.logit = tf.nn.softmax(final_dense, name='logit')
            self.classifer_loss = tf.losses.softmax_cross_entropy(self.labels, final_dense, label_smoothing=0.1, reduction=tf.losses.Reduction.MEAN)
            self.l2_loss = self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
            )
            total_loss = self.classifer_loss + self.l2_loss
            lossavg = tf.train.ExponentialMovingAverage(0.9, name='loss_moveavg')
            lossavg_op = lossavg.apply([total_loss])
            with tf.control_dependencies([lossavg_op]):
                self.total_loss = tf.identity(total_loss)
            var_list = tf.trainable_variables()
            varavg = tf.train.ExponentialMovingAverage(0.9, name='var_moveavg')
            varavg_op = varavg.apply(var_list)
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
            train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([update_ops, lossavg_op, varavg_op, train_op])
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(final_dense, 1), tf.argmax(self.labels, 1)), tf.float32), name='accuracy'
            )

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def train_one_batch(self, images, labels, lr, sess=None):
        self.is_training = True
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        _, loss, acc = sess_.run([self.train_op, self.total_loss, self.accuracy],
                                 feed_dict={
                                     self.images: images,
                                     self.labels: labels,
                                     self.lr: lr
                                 })
        return loss, acc

    def validate_one_batch(self, images, labels, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        logit, acc = sess_.run([self.logit, self.accuracy], feed_dict={
                                     self.images: images,
                                     self.labels: labels,
                                     self.lr: 0.
                                 })
        return logit, acc

    def test_one_batch(self, images, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        logit = sess_.run([self.logit], feed_dict={
                                     self.images: images,
                                     self.lr: 0.
                                 })
        return logit

    def save_weight(self, mode, path, sess=None):
        assert(mode in ['latest', 'best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        saver = self.saver if mode == 'latest' else self.best_saver
        saver.save(sess_, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, mode, path, sess=None):
        assert(mode in ['latest', 'best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        saver = self.saver if mode == 'latest' else self.best_saver
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess_, path)
            print('load', mode, 'model in', path, 'successfully')
        else:
            raise FileNotFoundError('Not Found Model File!')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu, name=None):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            name=name
        )
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _group_conv(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        total_conv = []
        filters_per_path = filters // 32
        axes = 3 if self.data_format == 'channels_last' else 1
        for i in range(32):
            split_bottom = tf.gather(bottom, tf.range(i*32, (i+1)*32), axis=axes)
            conv = self._conv_bn_activation(split_bottom, filters_per_path, kernel_size, strides, activation)
            total_conv.append(conv)
        total_conv = tf.concat(total_conv, axis=axes)
        return total_conv

    def _basic_block(self, bottom, filters):
                conv = self._conv_bn_activation(bottom, filters, 3, 1)
                conv = self._conv_bn_activation(conv, filters, 3, 1)
                axis = 3 if self.data_format == 'channels_last' else 1
                input_channels = tf.shape(bottom)[axis]
                shutcut = tf.cond(
                    tf.equal(input_channels, filters),
                    lambda: bottom,
                    lambda: self._conv_bn_activation(bottom, filters, 1 ,1)
                )
                return conv + shutcut

    def _residual_bottleneck(self, bottom, filters):
                conv = self._conv_bn_activation(bottom, filters, 1, 1)
                if self.is_groupconv:
                    conv = self._group_conv(conv, filters, 3, 1)
                else:
                    conv = self._conv_bn_activation(conv, filters, 3, 1)
                conv = self._conv_bn_activation(conv, filters*4, 1, 1)
                shutcut = self._conv_bn_activation(bottom, filters*4, 1, 1)
                return conv + shutcut

    def _dla_generator(self, bottom, filters, levels, stack_block_fn):
        if levels == 0:
            block1 = stack_block_fn(bottom, filters)
            block2 = stack_block_fn(block1, filters)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 1, 1)
        else:
            block1 = self._dla_generator(bottom, filters, levels-1, stack_block_fn)
            block2 = self._dla_generator(block1, filters, levels-1, stack_block_fn)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 1, 1)
        return aggregation

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )


