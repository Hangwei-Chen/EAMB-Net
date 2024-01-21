import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import einops

class resnet_base(nn.Module):
    def __init__(self, loadweights=True):
        super(resnet_base, self).__init__()
        resnet50Pretrained = torchvision.models.resnet50(pretrained= loadweights)
        self.feature1 = nn.Sequential(*list(resnet50Pretrained.children())[:5])
        self.feature2 = list(resnet50Pretrained.children())[5]
        self.feature3 = list(resnet50Pretrained.children())[6]
        self.feature4 = list(resnet50Pretrained.children())[7]

    def forward(self, x):
        input_size = x.size()
        x = F.interpolate(x, size=[(input_size[2] // 2), (input_size[3] // 2)], mode="bilinear", align_corners=True)
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        return f1,f2,f3


class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.emo_types = 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.fc_layer = nn.Linear(256, self.emo_types, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f1, f2, f3):
        x = self.conv1(f3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f2
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f1
        x = self.conv3(x)
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        conf = F.softmax(logits, dim=1)
        with torch.no_grad():
            B,C,H,W = x.shape
            w  = self.fc_layer.weight.data # cls_num, channels
            trans_w = einops.repeat(w, 'n c -> b n c', b=B)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x) # b n hw
            cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
            cam = cam / (cam.max(dim=-1)[0].unsqueeze(-1) + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=H, w=W)
            eam = torch.sum(conf[:,:,None,None] * cam, dim=1, keepdim=True)

            # save_folder='./cam_figure/'
            # emotion_type=['Amu','Awe','Con','Exci','Anger','Disg','Fear','Sad']
            # for i in range(8):
            #     cam_map = cam.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
            #     cam_map = cam_map[:,:,i]
            #     norm_cam = cv2.normalize(cam_map, None, 0, 255, cv2.NORM_MINMAX)
            #     norm_cam = np.asarray(norm_cam, dtype=np.uint8)
            #     heat_cam = cv2.applyColorMap(norm_cam, cv2.COLORMAP_JET)
            #     cv2.imwrite(os.path.join(save_folder, emotion_type[i]+ ".jpg"), heat_cam)
            # kcm_map = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
            # norm_kcm = cv2.normalize(kcm_map, None, 0, 255, cv2.NORM_MINMAX)
            # norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
            # heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join(save_folder,  "eam.jpg"), heat_im)
            # kcm = F.interpolate(kcm, scale_factor=4, mode='bilinear', align_corners=True)

            return logits, cam, eam, conf

class EmoClassifier(nn.Module):
    def __init__(self, loadweights=True):
        super(EmoClassifier, self).__init__()
        self.backbone   = resnet_base(loadweights=loadweights)
        self.emotion_module = EmotionModel()

    def forward(self, x, only_classify=False):
        f1,f2,f3 = self.backbone(x)
        logits,cam,eam,conf = self.emotion_module(f1,f2,f3)
        return logits,cam, eam,conf

