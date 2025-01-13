from .finetuning_model import FinetuningModel
from torch import nn
from core.utils import accuracy
import torch


class CLDFD_PreTrain(FinetuningModel):
    def __init__(self, feat_dim, num_class, **kwargs):
        super(CLDFD_PreTrain, self).__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(feat_dim, num_class)
        self.args = kwargs

    def set_forward_loss(self, batch):
        """
        param: batch
        return:
        """

        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        return 0, 0

        # image, global_target = batch
        # image = image.to(self.device)
        # global_target = global_target.to(self.device)
        # with torch.no_grad():
        #     feat = self.emb_func(image)
        # output = self.classifier(feat)
        # acc = accuracy(output, global_target.reshape(-1))
        # return output, acc

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        """
        :param support_feat:
        :param support_target:
        :param query_feat:
        :return:
        """
        pass
