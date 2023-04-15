#%%
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to the frozen inference graph and label map
PATH_TO_FROZEN_GRAPH = '/Users/marcus/models/research/object_detection/export_inference_graph.py'
PATH_TO_LABEL_MAP = 'path/to/label_map.pbtxt'

# Load the TensorFlow model
def load_model(path_to_frozen_graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# Load label map
def load_label_map(path_to_label_map, num_classes):
    label_map = label_map_util.load_labelmap(path_to_label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

# Run object detection on a video frame
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes, num_detections

def main():
    # Load model and label map
    detection_graph = load_model(PATH_TO_FROZEN_GRAPH)
    category_index = load_label_map(PATH_TO_LABEL_MAP, num_classes=90)

    # Start video stream
    video_url = 'your_video_stream_link'
    cap = cv2.VideoCapture(video_url)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects in the video frame
                boxes, scores, classes, num_detections = detect_objects(frame, sess, detection_graph)

                # Visualize the detection results on the video frame
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Display the resulting frame
                cv2.imshow('Live Stream Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

   

# %%
