import unittest

from object_detection.lib.box_coders.yolo_box_coder import YoloBoxCoder

import tensorflow as tf

class TestYoloBoxCoder(unittest.TestCase):

    def test_boxCoder(self):
        grid_wh = tf.constant([14, 14], dtype=tf.float32)
        box_coder = YoloBoxCoder(grid_wh)
        anchors = tf.constant([[128, 64], [64, 32], [32, 16]], shape=(3, 2), dtype=tf.float32)
        boxes = tf.constant(
            [
                [0, 0, 30, 20, 1],
                [15, 12, 15+130, 12+60, 2],
                [103, 143, 103+70, 143+40, 3]
            ],
            shape=(1, 3, 5),
            dtype=tf.float32)
        img_wh = tf.constant([256, 256], dtype=tf.float32)
        anchors /= img_wh
        boxes_x1y1 = boxes[..., 0:2] / img_wh
        boxes_x2y2 = boxes[..., 2:4] / img_wh
        boxes_cls = tf.expand_dims(boxes[..., 4], -1)
        boxes = tf.concat([boxes_x1y1, boxes_x2y2, boxes_cls], -1)
        encoded = box_coder.encode(boxes, anchors) # (batch_size, grid_y, grid_x, anchors, (tx, ty, tw, ty, obj, cls))
        decoded = box_coder.decode(encoded, anchors) # (batch_size, grid_y, grid_x, anchors, (x_min, y_min, x_max, y_max, obj, cls))
        pboxes = decoded[..., 0:4]
        pobj = decoded[..., 4]
        pcls = tf.expand_dims(decoded[..., 5], -1)

        batch_size = tf.shape(boxes)[0]
        num_boxes = tf.shape(boxes)[1]
        num_anchors = tf.shape(anchors)[0]
        grid_w = int(grid_wh[0])
        grid_h = int(grid_wh[1])

        # Collect all boxes and classness that the obj socre is grater than 0.5.
        results = []
        for n in range(batch_size):
            for i in range(grid_h):
                for j in range(grid_w):
                    for k in range(num_anchors):
                        if pobj[n][i][j][k] > 0.5:
                            results.append(tf.concat([pboxes[n][i][j][k], pcls[n][i][j][k]], -1))

        self.assertEqual(len(results), num_boxes)

        # Make sure no differences between GT and decoded results.
        # GT and decoded results are normalized.
        boxes = tf.reshape(boxes, (batch_size * num_boxes, -1))
        for i in range(batch_size * num_boxes):
            diff = tf.reduce_sum(results[i] - boxes[i])
            self.assertAlmostEqual(diff, 0, delta=1e-4)
