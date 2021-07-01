import unittest

from object_detection.lib.box_coders.utils import argmax_bbox_anchor_iou
from object_detection.lib.box_coders.utils import transform_bbox_to_grided_bbox

import tensorflow as tf

class TestAnchorMatcher(unittest.TestCase):

    def test_best_anchor_idx(self):
        anchors = tf.constant([[128, 64], [64, 32], [32, 16]], shape=(3, 2), dtype=tf.float32)
        boxes = tf.constant(
            [
                [0, 0, 30, 20, 1],
                [15, 12, 15+130, 12+60, 2],
                [103, 143, 103+70, 143+40, 3]
            ],
            shape=(1, 3, 5),
            dtype=tf.float32)
        best_anchor = argmax_bbox_anchor_iou(boxes, anchors)
        self.assertEqual(best_anchor[0][0], 2)
        self.assertEqual(best_anchor[0][1], 0)
        self.assertEqual(best_anchor[0][2], 1)

class TestTransformBoxToGridedBox(unittest.TestCase):

    def test_equal_WH(self):
        anchors = tf.constant([[128, 64], [64, 32], [32, 16]], shape=(3, 2), dtype=tf.float32)
        boxes = tf.constant(
            [
                [0, 0, 30, 20, 1],
                [15, 12, 15+130, 12+60, 2],
                [103, 143, 103+70, 143+40, 3]
            ],
            shape=(1, 3, 5),
            dtype=tf.float32)
        img_wh = tf.constant([224, 224], dtype=tf.float32)
        grid_wh = img_wh // 8
        anchors /= img_wh
        boxes_x1y1 = boxes[..., 0:2] / img_wh
        boxes_x2y2 = boxes[..., 2:4] / img_wh
        boxes_cls = tf.expand_dims(boxes[..., 4], -1)
        boxes = tf.concat([boxes_x1y1, boxes_x2y2, boxes_cls], -1)
        grided_boxes = transform_bbox_to_grided_bbox(boxes, anchors, grid_wh)
        boxes_xy = (boxes[..., 0:2] + boxes[..., 2:4]) / 2
        grid_xy = tf.cast(boxes_xy // (1/grid_wh), tf.int32)

        def _get_grid_item(batch_id, box_id):
            x = grid_xy[batch_id][box_id][0]
            y = grid_xy[batch_id][box_id][1]
            return grided_boxes[batch_id][y][x]

        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 0)[0], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 0)[1], -1).numpy(), 0)
        self.assertGreater(tf.reduce_sum(_get_grid_item(0, 0)[2], -1).numpy(), 0)
        self.assertEqual(_get_grid_item(0, 0)[2].numpy()[4], 1) # objectness
        self.assertEqual(_get_grid_item(0, 0)[2].numpy()[5], 1) # class id

        self.assertGreater(tf.reduce_sum(_get_grid_item(0, 1)[0], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 1)[1], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 1)[2], -1).numpy(), 0)
        self.assertEqual(_get_grid_item(0, 1)[0].numpy()[4], 1) # objectness
        self.assertEqual(_get_grid_item(0, 1)[0].numpy()[5], 2) # class id

        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 2)[0], -1).numpy(), 0)
        self.assertGreater(tf.reduce_sum(_get_grid_item(0, 2)[1], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 2)[2], -1).numpy(), 0)
        self.assertEqual(_get_grid_item(0, 2)[1].numpy()[4], 1) # objectness
        self.assertEqual(_get_grid_item(0, 2)[1].numpy()[5], 3) # class id

    def test_not_equal_WH(self):
        anchors = tf.constant([[128, 64], [64, 32], [32, 16]], shape=(3, 2), dtype=tf.float32)
        boxes = tf.constant(
            [
                [0, 0, 30, 20, 1],
                [15, 12, 15+130, 12+60, 2],
                [103, 143, 103+70, 143+40, 3]
            ],
            shape=(1, 3, 5),
            dtype=tf.float32)
        img_wh = tf.constant([448, 224], dtype=tf.float32)
        grid_wh = img_wh // 8
        anchors /= img_wh
        boxes_x1y1 = boxes[..., 0:2] / img_wh
        boxes_x2y2 = boxes[..., 2:4] / img_wh
        boxes_cls = tf.expand_dims(boxes[..., 4], -1)
        boxes = tf.concat([boxes_x1y1, boxes_x2y2, boxes_cls], -1)
        grided_boxes = transform_bbox_to_grided_bbox(boxes, anchors, grid_wh)
        boxes_xy = (boxes[..., 0:2] + boxes[..., 2:4]) / 2
        grid_xy = tf.cast(boxes_xy // (1/grid_wh), tf.int32)

        def _get_grid_item(batch_id, box_id):
            x = grid_xy[batch_id][box_id][0]
            y = grid_xy[batch_id][box_id][1]
            return grided_boxes[batch_id][y][x]

        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 0)[0], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 0)[1], -1).numpy(), 0)
        self.assertGreater(tf.reduce_sum(_get_grid_item(0, 0)[2], -1).numpy(), 0)
        self.assertEqual(_get_grid_item(0, 0)[2].numpy()[4], 1) # objectness
        self.assertEqual(_get_grid_item(0, 0)[2].numpy()[5], 1) # class id

        self.assertGreater(tf.reduce_sum(_get_grid_item(0, 1)[0], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 1)[1], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 1)[2], -1).numpy(), 0)
        self.assertEqual(_get_grid_item(0, 1)[0].numpy()[4], 1) # objectness
        self.assertEqual(_get_grid_item(0, 1)[0].numpy()[5], 2) # class id

        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 2)[0], -1).numpy(), 0)
        self.assertGreater(tf.reduce_sum(_get_grid_item(0, 2)[1], -1).numpy(), 0)
        self.assertEqual(tf.reduce_sum(_get_grid_item(0, 2)[2], -1).numpy(), 0)
        self.assertEqual(_get_grid_item(0, 2)[1].numpy()[4], 1) # objectness
        self.assertEqual(_get_grid_item(0, 2)[1].numpy()[5], 3) # class id
