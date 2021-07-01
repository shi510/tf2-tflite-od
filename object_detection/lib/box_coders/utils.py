import tensorflow as tf

def get_best_anchor_idx(boxes, anchors):
    """Calculate best matched anchor index.

    Args:
        boxes: (batch_size, num_boxes, (x_min, y_min, x_max, y_max, cls))
        anchors: (num_anchors, (w, h))

    Returns:
        Best matched anchor index.
        (batch_size, num_boxes)
    """
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = boxes[..., 2:4] - boxes[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
            tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.argmax(iou, axis=-1)
    return anchor_idx
