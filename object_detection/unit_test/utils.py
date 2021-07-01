import unittest

from object_detection.lib.box_coders.utils import get_best_anchor_idx

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
        
        best_anchor = get_best_anchor_idx(boxes, anchors)
        self.assertEqual(best_anchor[0][0], 2)
        self.assertEqual(best_anchor[0][1], 0)
        self.assertEqual(best_anchor[0][2], 1)
