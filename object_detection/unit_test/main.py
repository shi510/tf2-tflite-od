import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest

from object_detection.unit_test.utils_test import *
from object_detection.unit_test.yolo_box_coder_test import *

if __name__ == '__main__':
    unittest.main()
