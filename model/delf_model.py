from model.delf_v1 import DelfV1
from model.delf_v2 import DelfV2
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from nets.nasnet import nasnet
slim = tf.contrib.slim

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
        bounding_boxes = tf.reshape(tf.constant([0., 0., 1., 1.]), [1, 1, 4])
        def _get_locations(box):
            return tf.divide(
                tf.add(
                    tf.gather(box, [0, 1], axis=1), tf.gather(box, [2, 3], axis=1)),
                2.0)

        def random_crop(image):
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bounding_boxes)
            distorted_image = tf.image.crop_to_bounding_box(
                image, begin[0], begin[1], size[0], size[1])
            distorted_image = tf.image.resize_images(
                distorted_image, tf.constant([224, 224]))
            return distorted_image

        X1 = tf.map_fn(random_crop, X1)

        # get feature map from resnet
        delf_model = DelfV1(resnet_v, skipcon_attn=skipcon_attn)

        # get logits, features and attentions from delf model
        logits_1, attn_1, feat_1 = delf_model.AttentionModel(X1, num_classes, training_resnet=True, training_attention=True)
        logits_2, attn_2, feat_2 = delf_model.AttentionModel(X2, num_classes, training_resnet=True, training_attention=True, reuse=True)

        def _ModelFn(images, normalized_image, reuse):
            """Attention_model to get feature map and attention score map.

            Args:
            images: Image tensor.
            normalized_image: Whether or not the images are normalized.
            reuse: Whether or not the layer and its variables should be reused.
            Returns:
            attention: Attention score after the non-linearity.
            feature_map: Feature map after ResNet convolution.
            """
            if normalized_image:
                image_tensor = images
            else:
                image_tensor = NormalizePixelValues(images)

            # Extract features and attention scores.
            _, attention, _, feature_map, _ = delf_model.GetAttentionPrelogit(
                image_tensor,
                attention_nonlinear='softplus',
                attention_type='use_l2_normalized_feature',
                kernel=[1, 1],
                training_resnet=False,
                training_attention=True,
                reuse=True)
            return attention, feature_map

        def _ExtractKeypointDescriptor(image):
            boxes, feature_scales, features, scores = (
                feature_extractor.ExtractKeypointDescriptor(
                    image,
                    layer_name='resnet_v1_50/block3',
                    image_scales=tf.constant([0.25, 0.3536, 0.5000, 0.7072, 1.0]),
                    iou=1.0,
                    max_feature_num=10,
                    abs_thres=1.5,
                    model_fn=_ModelFn))
            return boxes, feature_scales, features, scores

        # get descriptors
        boxes, feature_scales, features, scores = tf.map_fn(
            _ExtractKeypointDescriptor, 
            X1, 
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32))

        locations = tf.map_fn(_get_locations, boxes)
        attentions = tf.map_fn(lambda x: tf.reshape(x, [tf.shape(x)[0]]), scores)

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
        self.features = tf.add(features, 0, name="features")
        self.similarity = tf.expand_dims(sim_12, axis=-1)
        self.boxes = tf.add(boxes, 0, name="boxes")
        self.scales = tf.add(feature_scales, 0, name="scales")
        self.scores = tf.add(attentions, 0, name="scores")
        self.locations = tf.add(locations, 0, name="locations")


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


class Nasnet(object):
    def __init__(self, X1, X2, num_classes, skipcon_attn=False,
                 stop_gradient_sim=False, logit_concat_sim=False):

        def get_nasnet(inputs, num_classes, reuse=True): 
            with tf.variable_scope('Nasnet_block', [inputs], reuse=tf.AUTO_REUSE) as sc:
                logits, end_points = nasnet.build_nasnet_large(inputs,
                                                               num_classes)
                return logits, end_points

        with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
            logits_1, end_points_1 = get_nasnet(X1, num_classes)
            features_1 = end_points_1['global_pool']

        with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
            logits_2, end_points_2 = get_nasnet(X2, num_classes)
            features_2 = end_points_2['global_pool']

        if skipcon_attn:
            features_1 += tf.reduce_sum(features_1, [1, 2])
            features_2 += tf.reduce_sum(features_2, [1, 2])

        logits_1_support = logits_1
        logits_2_support = logits_2

        if stop_gradient_sim:
            features_1 = tf.stop_gradient(features_1)
            features_2 = tf.stop_gradient(features_2)
            logits_1_support = tf.stop_gradient(logits_1_support)
            logits_2_support = tf.stop_gradient(logits_2_support)

        def dual_vector_fc(feat_x, reuse=None):
            with tf.variable_scope("dual_vector_fc_v1") as scope:
                v = fully_connected(feat_x, 1024, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
            with tf.variable_scope("dual_vector_fc_v2") as scope:
                v = fully_connected(v, 512, activation_fn=None, scope=scope, reuse=reuse)
            return v

        if logit_concat_sim:
            feat_attn_1 = tf.concat([features_1, logits_1_support], axis=-1)
            feat_attn_2 = tf.concat([features_2, logits_2_support], axis=-1)

        # pass through 2 dense layer (feature -> 1024 -> 512 > output)
        feat_attn_1 = dual_vector_fc(features_1)
        feat_attn_2 = dual_vector_fc(features_2, reuse=True)

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
        self.feat_map = tf.add(features_1, 0, name="feature_map")
        self.similarity = tf.expand_dims(sim_12, axis=-1)

