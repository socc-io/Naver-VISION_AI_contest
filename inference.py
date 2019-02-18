import tensorflow as tf
import numpy as np
from data_loader import image_generator, query_expand_generator

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io

from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform

def get_cnt_inliers(que_outputs, ref_outputs):

    locations_1, descriptors_1 = que_outputs
    locations_2, descriptors_2 = ref_outputs
    num_features_1 = locations_1.shape[0]
    num_features_2 = locations_2.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([ 
        locations_2[i,] 
        for i in range(num_features_2) 
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    print(inliers)
    if inliers is None:
        inliers = [0]
    return sum(inliers)

def get_feature(queries, references, sess):

    query_dataset = tf.data.Dataset.from_generator(
        lambda:image_generator(queries),
        output_types=tf.float32,
        output_shapes=[224, 224, 3]).batch(1)

    reference_dataset = tf.data.Dataset.from_generator(
        lambda:image_generator(references),
        output_types=tf.float32,
        output_shapes=[224, 224, 3]).batch(1)

    query_iterator = query_dataset.make_initializable_iterator()
    reference_iterator = reference_dataset.make_initializable_iterator()
    query_img = query_iterator.get_next()
    reference_img = reference_iterator.get_next()

    sess.run([query_iterator.initializer, reference_iterator.initializer])

    # Inference
    total_query_locations = []
    total_query_descriptors = []
    total_reference_locations = []
    total_reference_descriptors = []
    graph = tf.get_default_graph()
    feature_vector = graph.get_tensor_by_name("feature_vector:0")
    input_x = graph.get_tensor_by_name("input_X1:0")
    locations = graph.get_tensor_by_name('locations:0')
    descriptors = graph.get_tensor_by_name('features:0')
    feature_scales = graph.get_tensor_by_name('scales:0')
    attention = graph.get_tensor_by_name('scores:0')

    processed_query_num = 0
    processed_reference_num = 0
    while True:
        try:
            query_imgs = sess.run(query_img)
            feed_dict = {input_x: query_imgs}
            query_locations, query_descriptors, feature_scales_out, attention_out = sess.run(
                [locations, descriptors, feature_scales, attention],
                 feed_dict)
            total_query_locations.extend(query_locations)
            total_query_descriptors.extend(query_descriptors)
            processed_query_num += 1
        except tf.errors.OutOfRangeError:
            print("query[%d/%d] inference complete" % (processed_query_num, len(queries)))
            break

    while True:
        try:
            reference_imgs = sess.run(reference_img)
            feed_dict = {input_x: reference_imgs}
            reference_locations, reference_descriptors, feature_scales_out, attention_out = sess.run(
                [locations, descriptors, feature_scales, attention], 
                feed_dict)
            total_reference_locations.extend(reference_locations)
            total_reference_descriptors.extend(reference_descriptors)
            processed_reference_num += 1
        except tf.errors.OutOfRangeError:
            print("reference[%d/%d] inference complete" % (processed_reference_num, len(references)))
            break

    total_query_outputs = list(zip(total_query_locations, total_query_descriptors))
    total_reference_outputs = list(zip(total_reference_locations, total_reference_descriptors))
    return queries, total_query_outputs, references, total_reference_outputs

def query_expanded_get_feature(queries, references, sess, batch_size):

    query_dataset = tf.data.Dataset.from_generator(
        lambda:query_expand_generator(queries),
        output_types=tf.float32,
        output_shapes=[4, 224, 224, 3]).batch(batch_size)

    reference_dataset = tf.data.Dataset.from_generator(
        lambda:image_generator(references),
        output_types=tf.float32,
        output_shapes=[224, 224, 3]).batch(batch_size)

    query_iterator = query_dataset.make_initializable_iterator()
    reference_iterator = reference_dataset.make_initializable_iterator()
    query_img = query_iterator.get_next()
    reference_img = reference_iterator.get_next()
    sess.run([query_iterator.initializer, reference_iterator.initializer])

    # Inference
    total_query_vecs = []
    total_reference_vecs = []
    graph = tf.get_default_graph()
    feature_vector = graph.get_tensor_by_name("feature_vector:0")
    input_x = graph.get_tensor_by_name("input_X1:0")

    while True:
        try:
            query_imgs = sess.run(query_img)
            expanded_query_vecs = [
                sess.run(feature_vector, feed_dict = {input_x:query})
                for query in query_imgs]
            total_query_vecs.extend(expanded_query_vecs)
        except tf.errors.OutOfRangeError:
            print("query[%d/%d] inference complete" % (len(total_query_vecs), len(queries)))
            break

    while True:
        try:
            reference_imgs = sess.run(reference_img)
            feed_dict = {input_x: reference_imgs}
            reference_vecs = sess.run(feature_vector, feed_dict)
            total_reference_vecs.extend(reference_vecs)
        except tf.errors.OutOfRangeError:
            print("reference[%d/%d] inference complete" % (len(total_reference_vecs), len(references)))
            break

    total_query_vecs = np.asarray(total_query_vecs)
    total_reference_vecs = np.asarray(total_reference_vecs)
    return queries, total_query_vecs, references, total_reference_vecs
