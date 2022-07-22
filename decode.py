import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3, 4, 5], [0, 1, 2]]):
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []

        # The input is one of the three output matrix from the YOLO neck.
        # batch_size, (num_class + 4 + 1) * num_anchors, 13, 13
        # batch_size, (num_class + 4 + 1) * num_anchors, 26, 26
        # batch_size, (num_class + 4 + 1) * num_anchors, 52, 52
        for i, input in enumerate(inputs):
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # if the input image is 416 * 416,
            # stride_h = stride_w = 32, 16, 8
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            # scaled_anchors: The anchors size on the feature image.
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

            # reshape the input to get the prediction
            # batch_size, num_anchors, input_height, input_width, num_classes + 4 + 1
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # Using Sigmoid function to calculate the bbox center adjustment parameters
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            # Calculate the bbox height and width adjustment parameters:
            w = prediction[..., 2]
            h = prediction[..., 3]

            # the confidence to determine whether an object in the bbox
            conf = torch.sigmoid(prediction[..., 4])

            # the confidence for each class
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # Generate the grid, the center coordinates of the anchors
            # batch_size, num_anchors, 13, 13
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # Generate the height and the width for the anchors:
            # batch_size, num_anchors, 13, 13
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # Adjust the anchor based on the predictions' results:
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # Scale the output and wrap them up
            # output: (bounding_boxes_locations, has_object, class_likelihood)
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)

