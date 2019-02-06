from model.delf_v1 import DelfV1
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io


class Delf_model(object):
    def __init__(self, X, num_classes):
        delf_model = DelfV1('resnet_v1_50/block3')
        outputs = delf_model.AttentionModel(X, num_classes, training_resnet=True, training_attention=True)
        self.logits, self.attention, self.feature_map = outputs


class Delf_fuse_model(object):
    def __init__(self, X1, X2, num_classes, skipcon_attn=False, stop_gradient_sim=False):
        delf_model = DelfV1('resnet_v1_50/block3', skipcon_attn=skipcon_attn)
        logits_1, attn_1, feat_1 = delf_model.AttentionModel(X1, num_classes,
                   training_resnet=True, training_attention=True, reuse=True)
        logits_2, attn_2, feat_2 = delf_model.AttentionModel(X2, num_classes,
                   training_resnet=True, training_attention=True, reuse=True)


        feat_attn_1 = tf.reduce_sum(attn_1 * feat_1, [1,2])
        feat_attn_2 = tf.reduce_sum(attn_2 * feat_2, [1,2])

        if skipcon_attn :
            feat_attn_1 += tf.reduce_sum(feat_1, [1, 2])
            feat_attn_2 += tf.reduce_sum(feat_2, [1, 2])

        fuse_12 = tf.concat([
            feat_attn_1, feat_attn_2,
            feat_attn_1 - feat_attn_2, feat_attn_1 * feat_attn_2], axis=-1)

        if stop_gradient_sim:
            fuse_12 = tf.stop_gradient(fuse_12)
        fc_12 = fully_connected(fuse_12, 128, activation_fn=tf.nn.relu)
        sim_12 = fully_connected(fc_12, 1, activation_fn=None)

        self.feature_vector = tf.add(feat_attn_1, 0, name="feature_vector")
        self.logits_1 = tf.add(logits_1, name="logit_1")
        self.logits_2 = logits_2
        self.attention_1 = attn_1
        self.attention_2 = attn_2
        self.feature_map_1 = feat_1
        self.feature_map_2 = feat_2
        self.similarity = sim_12


class Delf_dual_model(object):

    def __init__(self, X1, X2, num_classes, skipcon_attn=False, stop_gradient_sim=False, logit_concat_sim=False):
        # get feature map from resnet
        delf_model = DelfV1('resnet_v1_50/block3', skipcon_attn=skipcon_attn)

        # get logits, features and attentions from delf model
        logits_1, attn_1, feat_1 = delf_model.AttentionModel(X1, num_classes, training_resnet=True, training_attention=True)
        logits_2, attn_2, feat_2 = delf_model.AttentionModel(X2, num_classes, training_resnet=True, training_attention=True, reuse=True)

        # apply trained attention on features
        feat_attn_1 = tf.reduce_sum(attn_1 * feat_1, [1, 2])
        feat_attn_2 = tf.reduce_sum(attn_2 * feat_2, [1, 2])
        
        # concate feature with feature_with_attn 
        if skipcon_attn:
            feat_attn_1 += tf.reduce_sum(feat_1, [1, 2])
            feat_attn_2 += tf.reduce_sum(feat_2, [1, 2])
        
        # fine tuning without attention
        logits_1_support = logits_1
        logits_2_support = logits_2

        if stop_gradient_sim:
            feat_attn_1 = tf.stop_gradient(feat_attn_1)
            feat_attn_2 = tf.stop_gradient(feat_attn_2)
            logits_1_support = tf.stop_gradient(logits_1_support)
            logits_2_support = tf.stop_gradient(logits_2_support)

        def dual_vector_fc(feat_attn_x, reuse=None):
            with tf.variable_scope("dual_vector_fc_v1") as scope:
                v = fully_connected(feat_attn_x, 1024, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
            with tf.variable_scope("dual_vector_fc_v2") as scope:
                v = fully_connected(v, 2048, activation_fn=None, scope=scope, reuse=reuse)
            return v

        if logit_concat_sim:
            feat_attn_1 = tf.concat([feat_attn_1, logits_1_support], axis=-1)
            feat_attn_2 = tf.concat([feat_attn_2, logits_2_support], axis=-1)

        # pass through 2 dense layer (feature -> 1024 -> 512 > output) 
        feat_attn_1 = dual_vector_fc(feat_attn_1)
        feat_attn_2 = dual_vector_fc(feat_attn_2, reuse=True)

        # normalize features
        normalize_a = tf.nn.l2_normalize(feat_attn_1, axis=1)
        normalize_b = tf.nn.l2_normalize(feat_attn_2, axis=1)

        # calculate similarity of features using cosine similarity
        sim_12 = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
        sim_12 = (sim_12 - 0.5) * 32 # to approach 1

        self.feature_vector = tf.add(feat_attn_1, 0, name="feature_vector")
        self.logits_1 = tf.add(logits_1, 0, name="logit_1")
        self.logits_2 = logits_2
        self.feat_attn_1 = feat_attn_1
        self.feat_attn_2 = feat_attn_2
        self.attention_1 = attn_1
        self.attention_2 = attn_2
        self.feat_map = tf.add(feat_1, 0, name="feature_map")
        self.similarity = tf.expand_dims(sim_12, axis=-1)


class Delf_dual_reshape_model(object):

    def __init__(self, X1, X2, num_classes, skipcon_attn=False, stop_gradient_sim=False, logit_concat_sim=False):
        # get feature map from resnet
        delf_model = DelfV1('resnet_v1_50/block3', skipcon_attn=skipcon_attn)

        # get logits, features and attentions from delf model
        logits_1, attn_1, feat_1 = delf_model.AttentionModel(X1, num_classes, training_resnet=True,
                                                             training_attention=True)
        logits_2, attn_2, feat_2 = delf_model.AttentionModel(X2, num_classes, training_resnet=True,
                                                             training_attention=True, reuse=True)

        # apply trained attention on features
        flat_shape = [-1, feat_1.shape[1] * feat_1.shape[2] * feat_1.shape[3]]
        feat_attn_1 = tf.reshape(attn_1 * feat_1, flat_shape)
        feat_attn_2 = tf.reshape(attn_2 * feat_2, flat_shape)

        # concate feature with feature_with_attn
        if skipcon_attn:
            feat_attn_1 += tf.reshape(feat_1, flat_shape)
            feat_attn_2 += tf.reshape(feat_2, flat_shape)

        # fine tuning without attention
        logits_1_support = logits_1
        logits_2_support = logits_2

        if stop_gradient_sim:
            feat_attn_1 = tf.stop_gradient(feat_attn_1)
            feat_attn_2 = tf.stop_gradient(feat_attn_2)
            logits_1_support = tf.stop_gradient(logits_1_support)
            logits_2_support = tf.stop_gradient(logits_2_support)

        def dual_vector_fc(feat_attn_x, reuse=None):
            with tf.variable_scope("dual_vector_fc_v1") as scope:
                v = fully_connected(feat_attn_x, 1024, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
            with tf.variable_scope("dual_vector_fc_v2") as scope:
                v = fully_connected(v, 512, activation_fn=None, scope=scope, reuse=reuse)
            return v

        if logit_concat_sim:
            feat_attn_1 = tf.concat([feat_attn_1, logits_1_support], axis=-1)
            feat_attn_2 = tf.concat([feat_attn_2, logits_2_support], axis=-1)

        # pass through 2 dense layer (feature -> 1024 -> 512 > output)
        feat_attn_1 = dual_vector_fc(feat_attn_1)
        feat_attn_2 = dual_vector_fc(feat_attn_2, reuse=True)

        # normalize features
        normalize_a = tf.nn.l2_normalize(feat_attn_1, axis=1)
        normalize_b = tf.nn.l2_normalize(feat_attn_2, axis=1)

        # calculate similarity of features using cosine similarity
        sim_12 = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
        sim_12 = (sim_12 - 0.5) * 32  # to approach 1

        self.feature_vector = tf.add(feat_attn_1, 0, name="feature_vector")
        self.logits_1 = tf.add(logits_1, 0, name="logit_1")
        self.logits_2 = logits_2
        self.feat_attn_1 = feat_attn_1
        self.feat_attn_2 = feat_attn_2
        self.attention_1 = attn_1
        self.attention_2 = attn_2
        self.feat_map = tf.add(feat_1, 0, name="feature_map")
        self.similarity = tf.expand_dims(sim_12, axis=-1)