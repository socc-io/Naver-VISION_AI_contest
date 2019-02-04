import tensorflow as tf
import numpy as np
from data_loader import image_generator

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io


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
    total_query_vecs = []
    total_reference_vecs = []
    graph = tf.get_default_graph()
    feature_vector = graph.get_tensor_by_name("feature_vector:0")
    input_x = graph.get_tensor_by_name("input_X1:0")

    try:
        while True:
            query_imgs = sess.run(query_img)
            feed_dict = {input_x: query_imgs}
            query_vecs = sess.run(feature_vector, feed_dict)

            reference_imgs = sess.run(reference_img)
            feed_dict = {input_x: reference_imgs}
            reference_vecs = sess.run(feature_vector, feed_dict)

            total_query_vecs.extend(query_vecs)
            total_reference_vecs.extend(reference_vecs)

    except tf.errors.OutOfRangeError:
        pass

    total_query_vecs = np.asarray(total_query_vecs)
    total_reference_vecs = np.asarray(total_reference_vecs)
    return queries, total_query_vecs, references, total_reference_vecs
