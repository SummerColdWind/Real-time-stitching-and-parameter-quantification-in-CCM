from .segmenter import Segmenter
from .skeleton import get_skeleton
from .point import get_points
from .instance import get_instance
from .common import split, dilated

import cv2
from pathlib import Path
import skimage.io as io
from numpy import ndarray


class Quantizer:
    def __init__(self):
        self.segmenter = Segmenter()
        self.cache = {
            'segments': None,
            'nodes': None,
            'binary': None,
            'skeleton': None,
        }

    @staticmethod
    def parse_input(input_):
        if isinstance(input_, (str, Path)):
            image = io.imread(input_)
        elif isinstance(input_, ndarray):
            image = input_.copy()
        else:
            raise TypeError

        return image

    def instantiation(self, input_):
        image = self.parse_input(input_)
        segments, nodes = [], []

        binary = self.segmenter([image])[0]
        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)
        skeleton = get_skeleton(binary, filter_threshold=64)
        blocks, _ = split(skeleton, split_skeleton=True)
        for block in blocks:
            points, end_points = get_points(block)
            instance = get_instance(binary, block, points, end_points)
            segments_, nodes_ = instance
            segments.extend(segments_)
            nodes.extend(nodes_)

        for i, segment in enumerate(segments):
            segment.index = i

        for i, node in enumerate(nodes):
            node.index = i

        self.cache['segments'] = segments
        self.cache['nodes'] = nodes
        self.cache['binary'] = binary
        self.cache['skeleton'] = skeleton

        return segments, nodes

    def quantize(self, input_):
        image = self.parse_input(input_)

        valid_area = cv2.countNonZero(image)
        segments, nodes = self.instantiation(image)

        total_length = sum(s.length for s in segments)
        total_points = len(nodes)

        # 单位转换 每384pixel代表0.4mm
        total_length = total_length / 384 * 0.4  # mm
        area = valid_area / 384 / 384 * 0.4 * 0.4  # mm2
        cnfl = total_length / area
        cnbd = total_points / area

        p = {
            'cnfl': cnfl,
            'cnbd': cnbd,
            'area': area,
        }
        return p

    def plot_skeleton(self, input_):
        if any(v is None for v in self.cache.values()):
            self.instantiation(input_)

        canvas = self.cache['skeleton'].copy()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        point_color = {
            'branch': (0, 255, 0),
            'end': (0, 0, 255),
        }

        for node in self.cache['nodes']:
            x, y = node.center
            cv2.circle(canvas, (x, y), 2, point_color[node.class_node], -1)

        return canvas




    @staticmethod
    def quantize_from_skeleton(input_):
        if isinstance(input_, (str, Path)):
            image = io.imread(input_)
        elif isinstance(input_, ndarray):
            image = input_.copy()
        else:
            raise TypeError

        valid_area = 384 * 384
        segments, nodes = [], []

        binary = dilated(image, iteration=3)
        skeleton = image
        blocks, _ = split(skeleton, split_skeleton=True)
        for block in blocks:
            points, end_points = get_points(block)
            instance = get_instance(binary, block, points, end_points)
            segments_, nodes_ = instance
            segments.extend(segments_)
            nodes.extend(nodes_)

        for i, segment in enumerate(segments):
            segment.index = i

        for i, node in enumerate(nodes):
            node.index = i

        total_length = sum(s.length for s in segments)
        total_points = len(nodes)

        # 单位转换 每384pixel代表0.4mm
        total_length = total_length / 384 * 0.4  # mm
        area = valid_area / 384 / 384 * 0.4 * 0.4  # mm2
        cnfl = total_length / area
        cnbd = total_points / area
        return cnfl, cnbd
