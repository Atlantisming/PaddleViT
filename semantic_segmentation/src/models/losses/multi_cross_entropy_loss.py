"""MultiCrossEntropyLoss Implement
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def multi_cross_entropy_loss(pred_list,
                             label,
                             num_classes=60,
                             weights=[1, 0.4, 0.4, 0.4, 0.4],
                             ignore_index=255):
    """MultiCrossEntropyLoss Function
    """
    label = paddle.reshape(label, [-1, 1]) # (b, h, w) -> (bhw, 1)
    label.stop_gradient = True
    loss = 0
    for i, pred in enumerate(pred_list):
        pred_i = paddle.transpose(pred, perm=[0, 2, 3, 1]) # (b,c,h,w) -> (b,h,w,c)
        pred_i = paddle.reshape(pred_i, [-1, num_classes]) # (b,h,w,c) -> (bhw, c)
        pred_i = F.softmax(pred_i, axis=1)
        loss_i = F.cross_entropy(pred_i, label, ignore_index=ignore_index)
        loss += weights[i]*loss_i
    return loss


class MultiCrossEntropyLoss(nn.Layer):
    """MultiCrossEntropyLoss
    """
    def __init__(self, config):
        super(MultiCrossEntropyLoss, self).__init__()
        self.num_classes = config.DATA.NUM_CLASSES
        self.weights = config.TRAIN.WEIGHTS
        self.ignore_index = config.TRAIN.IGNORE_INDEX

    def forward(self, logit, label):
        return multi_cross_entropy_loss(pred_list=logit,
                                        label=label,
                                        num_classes=self.num_classes,
                                        weights=self.weights,
                                        ignore_index=self.ignore_index)
