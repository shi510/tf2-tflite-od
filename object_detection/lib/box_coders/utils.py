import tensorflow as tf

def argmax_bbox_anchor_iou(boxes, anchors):
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
    return tf.argmax(iou, axis=-1, output_type=tf.int32)

@tf.function
def transform_bbox_to_grided_bbox(boxes, anchors, grid_wh):
    """Transform the bounding box into grid world.

    Args:
        boxes: (batch_size, num_boxes, (x_min, y_min, x_max, y_max, cls))
        anchors: (num_anchors, (w, h))
        grid_wh: [width, height]

    Returns:
        a tensor representing N anchor-encoded boxes of the format
        (batch_size, grid_y, grid_x, num_anchors, ((x_min, y_min, x_max, y_max, obj, cls))
    """
    best_anchor_indices = argmax_bbox_anchor_iou(boxes, anchors) # (batch_size, num_boxes)
    best_anchor_indices = tf.expand_dims(best_anchor_indices, -1)
    grid_wh = tf.cast(grid_wh, dtype=tf.int32)
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    num_anchors = tf.shape(anchors)[0]
    grided_bbox = tf.zeros((batch_size, grid_wh[1], grid_wh[0], num_anchors, 6), dtype=tf.float32)

    boxes_xy = (boxes[..., 0:2] + boxes[..., 2:4]) / 2
    grid_wh = tf.cast(grid_wh, tf.float32)
    grid_xy = tf.cast(boxes_xy // (1 / grid_wh), tf.int32) # (batch_size, num_boxes, (x, y))

    indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    for i in tf.range(batch_size):
        for j in tf.range(num_boxes):
            idx = i * num_boxes + j
            box = boxes[i][j][0:4]
            class_num = boxes[i][j][4]
            anchor_idx = best_anchor_indices[i][j][0]
            indices = indices.write(
                idx, [i, grid_xy[i][j][1], grid_xy[i][j][0], anchor_idx])
            updates = updates.write(
                idx, [box[0], box[1], box[2], box[3], 1, class_num])

    grided_bbox = tf.tensor_scatter_nd_update(
        grided_bbox, indices.stack(), updates.stack())
    return grided_bbox
