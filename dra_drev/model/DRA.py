import torch
import torch.nn as nn
import torch.nn.functional as F



class HolisticHead(nn.Module):
    def __init__(self, in_dim, dropout=0):
        """Global pooling + MLP on [B, C, N] patch features."""
        super(HolisticHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, C, N] -> [B, 1] (absolute score)."""
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)


class PlainHead(nn.Module):
    def __init__(self, in_dim, topk_rate=0.1):
        super(PlainHead, self).__init__()
        self.scoring = nn.Conv1d(in_dim, 1, kernel_size=1)
        self.topk_rate = topk_rate

    def forward(self, x):
        """x: [B, C, N] -> Conv1d score map -> top-k mean -> [B, 1]."""
        x = self.scoring(x)
        x = x.view(x.size(0), -1)
        topk = max(int(x.size(1) * self.topk_rate), 1)
        x = torch.topk(torch.abs(x), topk, dim=1)[0]
        x = torch.mean(x, dim=1, keepdim=True)
        return x


class CompositeHead(PlainHead):
    def __init__(self, in_dim, topk=0.1):
        super(CompositeHead, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU()
        )

    def forward(self, x, ref):
        """Residual vs averaged reference patches, then PlainHead scoring."""
        ref = torch.mean(ref, dim=0, keepdim=True)
        ref = ref.repeat(x.size(0), 1, 1)
        x = ref - x
        x = self.conv(x)
        x = super().forward(x)
        return x


class DRA(nn.Module):
    def __init__(self, cfg):
        super(DRA, self).__init__()
        self.cfg = cfg

        self.in_c = 1152
        self.holistic_head = HolisticHead(self.in_c)
        self.seen_head = PlainHead(self.in_c, self.cfg.topk)
        self.pseudo_head = PlainHead(self.in_c, self.cfg.topk)
        self.composite_head = CompositeHead(self.in_c, self.cfg.topk)

    def forward(self, feature, label, training=None):
        if training is not None:
            self.training = training

        pcd_pyramid = list()
        for i in range(self.cfg.total_heads):
            pcd_pyramid.append(list())

        ref_feature = feature[:self.cfg.nRef,:,:].permute(0, 2, 1)
        pcd_feature = feature[self.cfg.nRef:,:,:].permute(0, 2, 1)

        if self.training:
            normal_scores = self.holistic_head(pcd_feature)

            if pcd_feature.size(0) != label.size(0):
                print("\n[ERROR] Batch size mismatch!")
                print(f"feature batch = {pcd_feature.size(0)} | label batch = {label.size(0)}")
                print(f"feature.shape = {pcd_feature.shape}")
                print(f"label.shape = {label.shape}")
                pdb.set_trace()

            abnormal_scores = self.seen_head(pcd_feature[label != 2])

            dummy_scores = self.pseudo_head(pcd_feature[label != 1])

            comparison_scores = self.composite_head(pcd_feature, ref_feature)
        else:
            normal_scores = self.holistic_head(pcd_feature)
            abnormal_scores = self.seen_head(pcd_feature)
            dummy_scores = self.pseudo_head(pcd_feature)
            comparison_scores = self.composite_head(pcd_feature, ref_feature)
        for i, scores in enumerate([normal_scores, abnormal_scores, dummy_scores, comparison_scores]):
            pcd_pyramid[i].append(scores)
        for i in range(self.cfg.total_heads):
            pcd_pyramid[i] = torch.cat(pcd_pyramid[i], dim=1)
            pcd_pyramid[i] = torch.mean(pcd_pyramid[i], dim=1)
        return pcd_pyramid
