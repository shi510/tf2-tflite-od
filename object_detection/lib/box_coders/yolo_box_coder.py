from object_detection.core import box_coder
from object_detection.lib.box_coders.utils import transform_bbox_to_grided_bbox

import tensorflow as tf

class YoloBoxCoder(box_coder.BoxCoder):
    """YOLO box coder."""

    def __init__(self, grid_wh):
        """Constructor for YoloBoxCoder.
        Args:
            grid_wh: grid's width and height normalized by its width and height.
        """
        self.grid_wh = grid_wh
        self.grid_indices = tf.meshgrid(tf.range(grid_wh[1]), tf.range(grid_wh[0]))
        self.grid_indices = tf.stack(self.grid_indices, axis=-1)
        self.grid_indices = tf.expand_dims(self.grid_indices, axis=2)

    def _encode(self, boxes, anchors):
        """Encode a box collection with respect to anchor collection.

        Args:
            boxes: (batch_size, num_boxes, (x_min, y_min, x_max, y_max, cls))
            anchors: (num_anchors, (w, h))

        Returns:
            a tensor representing N anchor-encoded boxes of the format
            (batch_size, grid_y, grid_x, num_anchors, (tx, ty, tw, th, obj, cls))
            tx and ty are relative to the location of the grid cell.
        """
        grided_bbox = transform_bbox_to_grided_bbox(boxes, anchors, self.grid_wh)
        min_xy = grided_bbox[..., 0:2]
        max_xy = grided_bbox[..., 2:4]
        remainders = grided_bbox[..., 4:]
        cxy = (min_xy + max_xy) / 2
        wh = max_xy - min_xy
        txy = cxy * tf.cast(self.grid_wh, tf.float32) - tf.cast(self.grid_indices, tf.float32)
        twh = tf.math.log(wh / anchors)
        twh = tf.where(tf.math.is_inf(twh), tf.zeros_like(twh), twh)
        return tf.concat([txy, twh, remainders], -1)

    def _decode(self, coded_boxes, anchors):
        """Decode coded boxes.

        Args:
            coded_boxes: (batch_size, grid_y, grid_x, anchors, (tx, ty, tw, th, obj, cls))
            anchors: (anchors, (w, h))

        Returns:
            boxes: (batch_size, grid_y, grid_x, anchors, (x_min, y_min, x_max, y_max, cls))
        """
        txy = coded_boxes[..., 0:2]
        twh = coded_boxes[..., 2:4]
        remainders = coded_boxes[..., 4:]

        txy = (txy + tf.cast(self.grid_indices, tf.float32)) / tf.cast(self.grid_wh, tf.float32)
        twh = tf.exp(twh) * anchors

        box_x1y1 = txy - twh / 2
        box_x2y2 = txy + twh / 2
        return tf.concat([box_x1y1, box_x2y2, remainders], axis=-1)
