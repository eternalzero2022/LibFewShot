import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

from core.utils import accuracy  # 计算准确率的工具函数
from .finetuning_model import FinetuningModel  # 继承自少样本模型基础类
from .. import DistillKLLoss


class DistillLayer(nn.Module):
    """
    蒸馏层类，负责加载教师模型并为学生模型提供指导。
    """
    def __init__(
        self,
        emb_func,
        is_distill,
        emb_func_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        """
        从路径加载教师模型权重（如果启用蒸馏）。
        """
        new_model = None
        if is_distill and state_dict_path is not None:
            new_model = copy.deepcopy(model)
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            new_model.load_state_dict(model_state_dict)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        """
        蒸馏层的前向传播。
        """
        if self.emb_func is not None:
            output = self.emb_func(x)
            return output
        return None


class CLDFD(FinetuningModel):
    """
    Cross-Level Distillation and Feature Denoising (CLDFD) 模型。
    """
    def __init__(
        self,
        feat_dim,
        num_class,
        gamma=1,
        alpha=0,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        feature_denoising=None,
        **kwargs
    ):
        super(CLDFD, self).__init__(**kwargs)

        # 模型参数
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.gamma = gamma
        self.alpha = alpha
        self.is_distill = is_distill
        self.feature_denoising = feature_denoising

        # 分类器
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        # 蒸馏层
        self.distill_layer = DistillLayer(
            self.emb_func,
            self.is_distill,
            emb_func_path,
        )

    def set_forward(self, batch):
        """
        评估阶段的前向传播逻辑。

        :param batch: 包含输入图像和标签的元组 (images, global_targets)
        :return: 输出预测结果和分类准确率
        """
        images, global_targets = batch
        images = images.to(self.device)

        # Step 1: 特征提取
        # 从学生模型提取特征
        with torch.no_grad():
            features = self.emb_func(images)

        # 如果启用特征去噪，则对特征进行稀疏化处理
        if self.feature_denoising.get("enable", False):
            features = self.feature_denoising(features)

        # Step 2: 支撑集和查询集划分
        # 将批量数据分解为多个任务 (episode)，每个任务有支撑集和查询集
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            features, mode=1
        )

        episode_size = support_feat.size(0)  # 每个批次中的任务数
        output_list = []
        acc_list = []

        # Step 3: 逐任务处理
        for idx in range(episode_size):
            SF = support_feat[idx]  # 支撑集特征
            QF = query_feat[idx]  # 查询集特征
            ST = support_target[idx]  # 支撑集标签
            QT = query_target[idx]  # 查询集标签

            # Step 4: 支撑集上适应阶段
            classifier = self.set_forward_adaptation(SF, ST)

            # Step 5: 查询集预测
            # 归一化查询集特征并转换为 numpy 格式，供分类器使用
            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            # 利用分类器进行预测
            output = classifier.predict(QF)
            acc = metrics.accuracy_score(QT, output) * 100  # 计算准确率

            output_list.append(output)
            acc_list.append(acc)

        # 整理任务的预测结果和平均准确率
        output = np.stack(output_list, axis=0)
        acc = sum(acc_list) / episode_size
        return output, acc

    def set_forward_loss(self, batch):
        """
        训练阶段的前向传播。
        """
        # TODO: 实现训练阶段的逻辑
        # 提示：调用蒸馏层和学生模型的特征提取器，计算分类损失和蒸馏损失。
        pass

    def set_forward_adaptation(self, support_feat, support_target):
        """
        在支撑集上进行分类器的动态训练，适应少样本任务。

        :param support_feat: 支撑集特征 (support features)
        :param support_target: 支撑集标签 (support targets)
        :return: 在支撑集上训练好的分类器
        """
        # 初始化一个逻辑回归分类器
        classifier = LogisticRegression(
            penalty="l2",
            random_state=0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
        )

        # Step 1: 特征归一化
        # 归一化支撑集特征并转换为 numpy 格式，供分类器使用
        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        # Step 2: 分类器训练
        # 在支撑集特征和标签上训练分类器
        classifier.fit(support_feat, support_target)

        return classifier

    def kd_group_loss(self, teacher_features, student_features, old_student_features):
        """
        计算跨层知识蒸馏损失。
        """
        # TODO: 实现蒸馏损失的计算逻辑
        pass

    def feature_denoising(self, features):
        """
        对特征表示进行稀疏化（top-k 筛选）。
        """
        k = self.feature_denoising.get("top_k", 10)  # 保留的最大激活值个数
        batch_size, feature_dim = features.shape

        # 获取每个样本特征的 top-k 索引
        topk_indices = torch.topk(features, k, dim=-1).indices

        # 构建稀疏化掩码
        mask = torch.zeros_like(features)
        mask.scatter_(dim=-1, index=topk_indices, value=1)

        # 应用掩码保留 top-k 激活值
        return features * mask


