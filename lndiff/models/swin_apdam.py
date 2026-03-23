import torch
import torch.nn as nn
import timm
from timm.models import load_checkpoint
import os


class VisionEncoder(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', checkpoint_path=None):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool=''
        )
        self.embed_dim = self.backbone.num_features
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            load_checkpoint(self.backbone, checkpoint_path, strict=False)

    def forward(self, x):
        features = self.backbone(x)

        if features.ndim == 4:
            B, H, W, C = features.shape
            features = features.view(B, H * W, C)
        elif features.ndim == 3 and features.shape[1] == self.embed_dim:
            features = features.permute(0, 2, 1)

        return features


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.class_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.class_query, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        query = self.class_query.expand(B, -1, -1)
        attn_out, _ = self.attn(query, x, x)
        x = self.norm(query + self.dropout(attn_out))
        return x.squeeze(1)


class Swin_APDAM_Model(nn.Module):
    def __init__(self, num_classes=2, swin_path=None):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            model_name='swin_tiny_patch4_window7_224',
            checkpoint_path=swin_path
        )

        embed_dim = 768
        self.attn_pool = AttentionPooling(embed_dim, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image):
        img_feat = self.vision_encoder(image)
        img_pooled = self.attn_pool(img_feat)
        logits = self.classifier(img_pooled)
        return logits