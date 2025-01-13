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

        self.old_student = None

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
        训练阶段的前向传播，计算损失并进行梯度更新。
        """
        images, global_targets = batch
        images = images.to(self.device)

        # Step 1: 从学生模型提取特征
        student_features = self.emb_func(images)

        # Step 2: 使用 `old_student` 作为教师模型提取特征
        if self.old_student is not None:
            # 使用 `old_student` 获取教师模型的特征
            teacher_features = self.old_student(images)
        else:
            # 如果没有 `old_student`，则使用当前模型的蒸馏层作为教师
            teacher_features = self.distill_layer(images)

        # Step 3: 计算蒸馏损失 (蒸馏的任务)
        distill_loss = self.kd_group_loss(teacher_features, student_features)

        # Step 4: 计算分类损失
        classification_loss = self.ce_loss_func(student_features, global_targets)

        # Step 5: 组合总损失
        total_loss = classification_loss + self.gamma * distill_loss

        # 更新 old_student 为当前的学生模型（深拷贝）
        self.old_student = copy.deepcopy(self.emb_func)

        return total_loss

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

    def kd_group_loss(self, teacher_features, student_features):
        """
        计算跨层知识蒸馏损失。
        """
        # Step 1: 定义蒸馏损失 (KL散度) 或其他相似度度量
        # 在此示例中，我们使用KL散度（可以更改为其他蒸馏损失策略）
        distill_loss = 0
        for l in range(len(teacher_features)):
            teacher_feat = teacher_features[l]
            student_feat = student_features[l]

            # Step 2: 对特征进行归一化处理
            teacher_feat = F.normalize(teacher_feat, p=2, dim=1)
            student_feat = F.normalize(student_feat, p=2, dim=1)

            # Step 3: 计算KL散度损失（可选其他损失形式）
            distill_loss += self.kl_loss_func(student_feat, teacher_feat)

        # Step 4: 返回总的蒸馏损失
        return distill_loss

    def feature_denoising(self, features):
        """
        应用特征去噪到最后一层的输出。
        """
        # TODO: 实现特征去噪的逻辑
        pass

