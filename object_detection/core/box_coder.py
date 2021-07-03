from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import six
import tensorflow as tf


class BoxCoder(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for box coder."""

    def encode(self, boxes, anchors):
        """Encode a box list relative to an anchor collection.

        Args:
            boxes: BoxList holding N boxes to be encoded
            anchors: BoxList of N anchors

        Returns:
            a tensor representing N relative-encoded boxes
        """
        with tf.name_scope('Encode'):
            return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        """Decode boxes that are encoded relative to an anchor collection.

        Args:
            rel_codes: a tensor representing N relative-encoded boxes
            anchors: BoxList of anchors

        Returns:
            boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
                with corners y_min, x_min, y_max, x_max)
        """
        with tf.name_scope('Decode'):
            return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        """Method to be overriden by implementations.

        Args:
            boxes: BoxList holding N boxes to be encoded
            anchors: BoxList of N anchors

        Returns:
            a tensor representing N relative-encoded boxes
        """
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        """Method to be overriden by implementations.

        Args:
            rel_codes: a tensor representing N relative-encoded boxes
            anchors: BoxList of anchors

        Returns:
            boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
                with corners y_min, x_min, y_max, x_max)
        """
        pass
