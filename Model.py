import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class FeatureExtractionHsi(nn.Module):
    def __init__(self):
        super(FeatureExtractionHsi, self).__init__()
        self.ge_w = nn.Conv2d(2, 6400, (3, 3), stride=(4, 4), padding=0)

        self.BN_1 = nn.BatchNorm2d(6400)

    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        """
        b, c, h, w = x_in.shape

        x_hsi = x_in

        x_avg = torch.mean(x_hsi, dim=1, keepdim=True)
        x_max, _ = torch.max(x_hsi, dim=1, keepdim=True)
        x_H = torch.cat([x_avg, x_max], dim=1)
        w_H = F.relu(self.BN_1(self.ge_w(x_H)))
        w_H = w_H.reshape(b, 100, 64, 3, 3)

        outputs = torch.zeros((b, 100, 11, 11)).cuda()
        for i in range(b):
            X = x_hsi[i].unsqueeze(0)
            W = w_H[i]
            output = F.conv2d(X, W, padding=1)
            outputs[i] = output

        return outputs


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.conv_h1 = nn.Conv2d(64, 100, (3, 3), stride=(1, 1), padding=1)
        self.BN_h1 = nn.BatchNorm2d(100)
        self.conv_l1 = nn.Conv2d(64, 100, (3, 3), stride=(1, 1), padding=1)
        self.BN_l1 = nn.BatchNorm2d(100)

        self.conv_p1 = nn.Conv2d(100, 64, (3, 3), stride=(1, 1), padding=1)
        self.BN_p1 = nn.BatchNorm2d(64)

        self.conv_h3 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1)
        self.BN_h3 = nn.BatchNorm2d(256)
        self.conv_h4 = nn.Conv2d(256, 100, (3, 3), stride=(1, 1), padding=1)
        self.BN_h4 = nn.BatchNorm2d(100)
        self.conv_l3 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1)
        self.BN_l3 = nn.BatchNorm2d(256)
        self.conv_l4 = nn.Conv2d(256, 100, (3, 3), stride=(1, 1), padding=1)
        self.BN_l4 = nn.BatchNorm2d(100)

    def forward(self, x_hsi, x_lidar):
        """
        x_in: [b,c,h,w]
        """
        x_h = F.relu(self.BN_h1(self.conv_h1(x_hsi)))
        x_l = F.relu(self.BN_l1(self.conv_l1(x_lidar)))

        x_h = F.relu(self.BN_p1(self.conv_p1(x_h)))
        x_l = F.relu(self.BN_p1(self.conv_p1(x_l)))

        x_H = torch.cat([x_h, x_hsi], dim=1)
        x_L = torch.cat([x_l, x_lidar], dim=1)

        x_H = F.relu(self.BN_h3(self.conv_h3(x_H)))
        x_L = F.relu(self.BN_l3(self.conv_l3(x_L)))
        x_H = F.relu(self.BN_h4(self.conv_h4(x_H)))
        x_L = F.relu(self.BN_l4(self.conv_l4(x_L)))

        return x_H, x_L


class FeatureFusion(nn.Module):
    def __init__(self, cls):
        super(FeatureFusion, self).__init__()
        self.conv_1 = nn.Conv2d(200, 128, (1, 1), stride=(1, 1), padding=0)
        self.conv_2 = nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=0)
        self.conv_3 = nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=0)

        self.BN_1 = nn.BatchNorm2d(128)
        self.BN_2 = nn.BatchNorm2d(64)
        self.BN_3 = nn.BatchNorm2d(32)

        self.FL = nn.Flatten()
        self.fc1 = nn.Linear((7*7*32), 128)
        self.fcr = nn.Linear(128, cls)

    def forward(self, h, l):
        """
        x_in: [b,c,h,w]
        """
        h_l = torch.cat([h, l], dim=1)
        x_F = F.relu(self.BN_1(self.conv_1(h_l)))
        x_F = F.relu(self.BN_2(self.conv_2(x_F)))
        x_F = F.relu(self.BN_3(self.conv_3(x_F)))

        x_FL_r = self.FL(x_F)
        x_FL = F.relu(self.fc1(x_FL_r))
        x_result = F.softmax((self.fcr(x_FL)), dim=1)

        return x_result, x_F, x_FL_r     # x_result为预测结果，x_F为融合特征，x_FL_r是用以对比学习的特征


class Model(nn.Module):
    def __init__(self, hsi_channel, lidar_channel, cls):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv2d(hsi_channel, 64, (1, 1), stride=(1, 1), padding=0)
        self.BN_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(lidar_channel, 64, (1, 1), stride=(1, 1), padding=0)
        self.BN_2 = nn.BatchNorm2d(64)

        self.FE = FeatureExtraction()

        self.FF = FeatureFusion(cls)

    def forward(self, h, l, h_ex, l_ex):
        """
        x_in: [b,h,w,c]
        """
        h = h.permute(0, 3, 1, 2)
        l = l.permute(0, 3, 1, 2)
        h_ex = h_ex.permute(0, 3, 1, 2)
        l_ex = l_ex.permute(0, 3, 1, 2)

        # embedding
        h = F.relu(self.BN_1(self.conv_1(h)))
        h_ex = F.relu(self.BN_1(self.conv_1(h_ex)))
        l = F.relu(self.BN_2(self.conv_2(l)))
        l_ex = F.relu(self.BN_2(self.conv_2(l_ex)))

        x_h, x_l = self.FE(h, l)
        x_hex, x_lex = self.FE(h_ex, l_ex)

        result_hl, F_HL, F_HL_con = self.FF(x_h, x_l)
        result_hexlex, F_HL_ex, F_HL_ex_con = self.FF(x_hex, x_lex)

        F_con = torch.cat([F_HL_con, F_HL_ex_con], dim=0)

        return result_hl, result_hexlex, F_HL, F_HL_ex, F_con
