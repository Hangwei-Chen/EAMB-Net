import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.EmotionNet import EmoClassifier


def emotion_model():
    emotion_model = EmoClassifier()
    emotion_model_path = './pretrain_emotion_model/emotion_model.pth'
    checkpoint_emotion_model = torch.load(emotion_model_path)
    emotion_model.load_state_dict(checkpoint_emotion_model['model'])
    return emotion_model


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SADEM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(SADEM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU()

        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU()
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)

        x = sim_map * x + sim_map * y + x

        return x


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class up_conv_bn_relu(nn.Module):
    def __init__(self, up_size, in_channels, out_channels, kernal_size=1, padding=0, stride=1):
        super(up_conv_bn_relu, self).__init__()
        self.upSample = nn.Upsample(size=(up_size, up_size), mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EAMBNet(nn.Module):
    def __init__(self):
        super(EAMBNet, self).__init__()
        self.emotion_model = emotion_model()
        for p in self.parameters():
            p.requires_grad = False
        self.model_x = torchvision.models.resnet50(pretrained=True)
        self.feature1_x = nn.Sequential(*list(self.model_x.children())[:5])
        self.feature2_x = list(self.model_x.children())[5]
        # self.feature3_x = list(self.model_x.children())[6]
        # self.feature4_x = list(self.model_x.children())[7]

        self.model_s = torchvision.models.resnet50(pretrained=True)
        self.feature1_s = nn.Sequential(*list(self.model_s.children())[:5])
        self.feature2_s = list(self.model_s.children())[5]
        self.feature3_s = list(self.model_s.children())[6]
        self.feature4_s = list(self.model_s.children())[7]

        self.up1 = up_conv_bn_relu(up_size=64, in_channels=2048, out_channels=256)
        self.CBR1 = conv_bn_relu(512, 56)
        self.CBR2 = conv_bn_relu(512, 56)
        self.CBR3 = conv_bn_relu(1024, 56)
        self.CBR4 = conv_bn_relu(2048, 56)
        self.CBR5 = conv_bn_relu(56, 256)
        self.CBR6 = conv_bn_relu(512, 256)
        self.CBR7 = conv_bn_relu(512, 256)

        self.SADEM1 = SADEM(56, 16)
        self.SADEM2 = SADEM(56, 16)
        self.CBAM = CBAMLayer(256)

        self.head = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        input_size = x.size()
        s = F.interpolate(x, size=[(input_size[2] // 2), (input_size[3] // 2)], mode="bilinear", align_corners=True)
        logits, cam, EAM, conf = self.emotion_model(x)
        x1 = self.feature1_x(x)  # 256, 128,128
        x2 = self.feature2_x(x1)  # 512,64,64
        # x3 = self.feature3_x(x2)# 1024,32,32
        # x4 = self.feature4_x(x3)

        s1 = self.feature1_s(s)  # 256, 64,64
        s2 = self.feature2_s(s1)  # 512,32,32
        s3 = self.feature3_s(s2)  # 1024,16,16
        s4 = self.feature4_s(s3)

        s2 = self.CBR1(s2)
        x2 = self.CBR2(x2)
        C = self.SADEM1(x2, s2)
        s3 = self.CBR3(s3)
        C = self.SADEM2(C, s3)
        C = self.CBR5(C)

        x4_ = self.up1(s4)
        cat = torch.cat((C, x4_), dim=1)
        cat = self.CBR6(cat)

        h_EAM = cat * EAM
        h_EAM = self.CBAM(h_EAM)
        Fusion_F = torch.cat((h_EAM, cat), dim=1)
        Fusion_F = self.CBR7(Fusion_F)
        score_feature = self.avgpool(Fusion_F).view(Fusion_F.size(0), -1)
        score = self.head(score_feature)
        return score


