# depth_feature_extractor.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Dict
import torch.nn.functional as F


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim=64, cfg=None):
        """
        根据 cfg["mode"] 控制是否使用 ResNet18 提取图像特征：
            - mode="resnet" 使用 CNN + MLP
            - mode="concat" 直接将展平图像与状态拼接输入 MLP
        """
        super().__init__(observation_space, features_dim)
        self.cfg = cfg or {}
        self.feature_extractor = self.cfg.get("feature_extractor", "concat")  # 默认使用 ResNet

        # === ResNet 图像分支 ===
        if self.feature_extractor == "resnet":
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]
            self.projector = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, cfg["resnet_output_dim"])
            )
            features_dim = cfg["resnet_output_dim"]
            self._features_dim = features_dim
        elif self.feature_extractor == "concat":
            # === 直接 flatten image ===
            image_flatten_dim = cfg["concat_output_dim"]  # 单通道图像（预设)
            features_dim = image_flatten_dim
            self._features_dim = features_dim
        elif self.feature_extractor == "mobilenet_v2":
            self.backbone = mobilenet_v2(weights='DEFAULT').features  # 去除分类层
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projector = nn.Sequential(
                nn.Linear(1280, 64),
                nn.ReLU(),
                nn.Linear(64, cfg["mobilenet_v2_output_dim"])
            )
            features_dim = cfg["mobilenet_v2_output_dim"]
            # === 更新特征维度 ===
            self._features_dim = features_dim

    def forward(self, observation):
        batch_image_feats = []

        for depth_img in observation:
            if self.feature_extractor in ["resnet", "mobilenet_v2"]:
                processed = self.preprocess_depth_to_3ch(depth_img)  # [1, 3, 224, 224]
                batch_image_feats.append(processed)
            elif self.feature_extractor == "concat":
                pooled = self.pool_depth_image_min_tensor(depth_img, grid_shape=(4, 4))  # [4, 4]
                flat = torch.as_tensor(pooled, dtype=torch.float32).flatten()
                batch_image_feats.append(flat)

        if self.feature_extractor == "resnet":
            x_img = torch.cat(batch_image_feats, dim=0)            # [B, 3, 224, 224]
            features = self.backbone(x_img)                         # [B, 512, 1, 1]
            features = features.view(x_img.size(0), -1)             # [B, 512]
            x_image = self.projector(features)                      # [B, resnet_output_dim]
        elif self.feature_extractor == "concat":
            x_image = torch.stack(batch_image_feats, dim=0)         # [B, 224*224]
        elif self.feature_extractor == "mobilenet_v2":
            x_img = torch.cat(batch_image_feats, dim=0) 
            feat = self.backbone(x_img)
            pooled = self.pool(feat)
            x_image = self.projector(pooled)

        return x_image # 最终特征输出

    def preprocess_depth_to_3ch(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        将 [1, 1, H, W] 深度图张量扩展为 3 通道，并归一化到 [-1, 1]。

        参数:
            depth_image: torch.Tensor，shape = [1, 1, H, W]，float32，值在 [0, 1]

        返回:
            depth_tensor: torch.Tensor，shape = [1, 3, H, W]，float32，值在 [-1, 1]
        """
        assert depth_image.dim() == 4 and depth_image.shape[1] == 1, \
            f"Expected shape [1, 1, H, W], got {depth_image.shape}"

        # 复制通道：从 [1, 1, H, W] → [1, 3, H, W]
        depth_3ch = depth_image.repeat(1, 3, 1, 1)

        # 归一化到 [-1, 1]
        depth_3ch = depth_3ch * 2.0 - 1.0

        return depth_3ch

    def pool_depth_image_min_tensor(self, depth_image: torch.Tensor, grid_shape=(4, 4)) -> torch.Tensor:
        """
        使用 PyTorch 对 2D 深度图进行最小值池化（min-pooling），按网格划分。
        
        参数:
            depth_image: torch.Tensor, shape=(H, W)，float32，取值范围通常在 [0, 1]
            grid_shape: tuple, (rows, cols)

        返回:
            pooled: torch.Tensor, shape=(rows, cols)，每个区域的最小深度
        """
        assert depth_image.dim() == 2, f"Expected 2D tensor, got {depth_image.shape}"
        
        H, W = depth_image.shape
        rows, cols = grid_shape
        h_step, w_step = H // rows, W // cols

        pooled = torch.empty((rows, cols), dtype=depth_image.dtype, device=depth_image.device)

        for i in range(rows):
            for j in range(cols):
                region = depth_image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                pooled[i, j] = torch.min(region)

        return pooled

