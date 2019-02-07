import tensorflow as tf
import numpy as np
from data_loader import image_generator

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
    locations_1 = np.squeeze(locations_1, axis=0)
    locations_2 = np.squeeze(locations_2, axis=0)
    descriptors_1 = np.squeeze(descriptors_1, axis=0)
    descriptors_2 = np.squeeze(descriptors_2, axis=0)
    num_features_1 = locations_1.shape[0]
    num_features_2 = locations_2.shape[0]
    print("1111")
    print(descriptors_1)
    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

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
    return sum(inliers)

def get_feature(queries, references, sess):
    queries = np.asarray(queries)
    references = np.asarray(references)
    query_dataset = tf.data.Dataset.from_generator(
        lambda:image_generator(queries),
        output_types=tf.float32,
        output_shapes=[224, 224, 3]).batch(32)

    reference_dataset = tf.data.Dataset.from_generator(
        lambda:image_generator(references),
        output_types=tf.float32,
        output_shapes=[224, 224, 3]).batch(32)

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


    try:
        while True:
            query_imgs = sess.run(query_img)
            feed_dict = {input_x: query_imgs}
            query_locations, query_descriptors, feature_scales_out, attention_out = sess.run(
                [locations, descriptors, feature_scales, attention],
                 feed_dict)

            reference_imgs = sess.run(reference_img)
            feed_dict = {input_x: reference_imgs}
            reference_locations, reference_descriptors, feature_scales_out, attention_out = sess.run(
                [locations, descriptors, feature_scales, attention], 
                feed_dict)

            total_query_locations.extend(query_locations)
            total_query_descriptors.extend(query_descriptors)
            total_reference_locations.extend(reference_locations)
            total_reference_descriptors.extend(reference_descriptors)
    except tf.errors.OutOfRangeError:
        pass

    total_query_outputs = (query_locations, query_descriptors)
    total_reference_outputs = (reference_locations, reference_descriptors)
    return queries, total_query_outputs, references, total_reference_outputs
