import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np


from core.utils import accuracy  # 计算准确率的工具函数
from .finetuning_model import FinetuningModel  # 继承自少样本模型基础类
from .. import DistillKLLoss
from core.data.collates.contrib.autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from core.trainer import Trainer


class Projector_SimCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''

    def __init__(self, in_dim = 512, out_dim = 512, mid_dim = None, bn = False):
        super(Projector_SimCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim if mid_dim is not None else self.in_dim
        if bn:
            layers = [nn.Linear(self.in_dim, self.mid_dim),
                      nn.BatchNorm1d(self.mid_dim),
                      nn.ReLU(inplace = True),
                      nn.Linear(self.mid_dim, self.out_dim)]
        else:
            layers = [nn.Linear(self.in_dim, self.mid_dim),
                      nn.ReLU(inplace=True),
                      nn.Linear(self.mid_dim, self.out_dim)]
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        x = self.projector(x)
        return x

class NTXentLoss(nn.Module):
    """
    用于 SimCLR 模型的负样本对损失计算。
    """
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 *
                    self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 *
                    self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(
            representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


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
    def forward(self, x, ret_layers=None):
        """
        蒸馏层的前向传播。
        """
        if self.emb_func is not None:
            output = self.emb_func(x, ret_layers)
            return output
        return None


class CLDFD(FinetuningModel):
    """
    Cross-Level Distillation and Feature Denoising (CLDFD.yaml) 模型。
    """
    def __init__(
        self,
        feat_dim,
        num_class,
        batch_size,
        gamma=1,
        alpha=0,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        feature_denoising=None,
        pic_size=224,
        transform_type="ImageNet",
        feature_dim=512,
        ss_proj_dim=128,
        temperature=1,
        epoch=60,
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
        self.pic_size = pic_size
        self.transform_type = transform_type
        self.epoch = epoch



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

        # 自监督损失
        self.feature_dim = feature_dim
        self.ss_proj_dim = ss_proj_dim
        self.temperature = temperature
        self.batch_size = batch_size
        self.simclr_proj = Projector_SimCLR(self.feature_dim, self.ss_proj_dim)
        self.simclr_criterion = NTXentLoss('cuda', self.batch_size, temperature = self.temperature, use_cosine_similarity = True)

        self.kd_proj0 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.kd_proj1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.kd_proj2 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

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
        训练阶段的前向传播，计算损失并进行梯度更新。
        """

        images, global_targets = batch
        images = images.to(self.device)

        # Transform step 1: 重新调整图像大小
        resize_transform = transforms.Resize((self.pic_size, self.pic_size))
        images = resize_transform(images)

        # Transform step 2: 定义数据增强策略
        argumentation_policy = None
        if(self.transform_type == "ImageNet"):
            argumentation_policy = ImageNetPolicy()
        elif(self.transform_type == "CIFAR10"):
            argumentation_policy = CIFAR10Policy()
        elif(self.transform_type == "SVHN"):
            argumentation_policy = SVHNPolicy()
        elif(self.transform_type == "SubPolicy"):
            argumentation_policy = SubPolicy()
        else:
            raise ValueError("Invalid transform_type. Choose from 'ImageNet', 'CIFAR10', 'SVHN', or 'SubPolicy'.")

        # Transform step 3: 将图像根据三种不同的策略进行增强，获得三份不同的增强后的图像
        X1 = argumentation_policy(images)
        X2 = argumentation_policy(images)
        X3 = argumentation_policy(images)

        X1 = X1.to(self.device)
        X2 = X2.to(self.device)
        X3 = X3.to(self.device)

        # Step 1: 从学生模型提取特征
        student_features, student_final = self.emb_func(X1,ret_layers=[4, 5, 6])
        student_features2, student_final2 = self.emb_func(X2, ret_layers=[4, 5, 6])

        # Step 1.5: 处理学生模型输出用于计算自适应损失
        z1_stu = self.simclr_proj(student_final)
        z2_stu = self.simclr_proj(student_final2)
        loss_sim = self.simclr_criterion(z1_stu, z2_stu)

        # Step 2: 使用教师模型提取特征
        with torch.no_grad():
            teacher_features, teacher_final = self.distill_layer(images, ret_layers=[5, 6, 7])

        # Step 3: 计算蒸馏损失 (蒸馏的任务)
        distill_loss = self.kd_group_loss(teacher_features, student_features, X3, epoch=Trainer.current_epoch)

        # Step 5: 组合总损失
        total_loss = loss_sim + self.gamma * distill_loss

        # 更新 old_student 为当前的学生模型（深拷贝）
        self.old_student = copy.deepcopy(self.emb_func)

        acc = accuracy(student_final, global_targets)

        return student_final, acc, total_loss

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

    def kd_group_loss(self, teacher_features, student_features, X3, epoch = 0):
        """
        计算跨层知识蒸馏损失。
        """
        if epoch == 0:
            loss_kd0 = F.mse_loss(self.kd_proj0(student_features[0]), teacher_features[0].detach())
            loss_kd1 = F.mse_loss(self.kd_proj1(student_features[1]), teacher_features[1].detach())
            loss_kd2 = F.mse_loss(self.kd_proj2(student_features[2]), teacher_features[2].detach())
        else:
            # 当前训练轮次除以总训练轮次
            momentum = epoch / self.epoch
            self.old_student.eval()
            with torch.no_grad():
                f1_old_map, _ = self.old_student(X3, ret_layers=[5, 6, 7])
                ft0 = (1 - momentum) * teacher_features[0].detach() + momentum * f1_old_map[0].detach()
                ft1 = (1 - momentum) * teacher_features[1].detach() + momentum * f1_old_map[1].detach()
                ft2 = (1 - momentum) * teacher_features[2].detach() + momentum * f1_old_map[2].detach()
            loss_kd0 = F.mse_loss(self.kd_proj0(student_features[0]), ft0.detach())
            loss_kd1 = F.mse_loss(self.kd_proj1(student_features[1]), ft1.detach())
            loss_kd2 = F.mse_loss(self.kd_proj2(student_features[2]), ft2.detach())
        return loss_kd0 + loss_kd1 + loss_kd2

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


