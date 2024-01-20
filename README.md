# EAMB-Net
This is the official code for the EAMBNet(TIM2024).

## 📎 Paper Link
- **Article title**: Image Aesthetics Assessment with Emotion-Aware Multi-Branch Network
- **Authors**: Hangwei chen, Feng Shao, Baoyang Mu, Qiuping Jinag
- **Institution**: Faculty of Electrical Engineering and Computer Science, Ningbo University
## 💡 Abstract
The aesthetic and appreciation of an image is the innate human perceptual ability. Emotion, as one of the most basic human perceptions, has been found to have a close relationship with aesthetics. However, explicitly incorporating the learned emotion cues into the image aesthetics assessment (IAA) model remains challenging. Additionally, humans consider both fine-grained details and holistic context information in aesthetic assessments. Therefore, the utilization of emotional information to enhance and modulate the representation of aesthetic features in context and detail is crucial for IAA. With this motivation, we propose a new IAA method named emotion-aware multi-branch network (EAMB-Net). Specifically, we first design two branches to extract aesthetic features related to detail and context. Then, an emotion branch is proposed to reveal the important emotion regions by generating the emotion-aware map (EAM). Finally, the EAM is further employed to infuse emotional knowledge into the aesthetic features and enhance the feature representation, producing the final aesthetic prediction. Experimental results validate that the proposed EAMB-Net can achieve superior performance in score regression, binary classification, and score distribution tasks, obtaining the classification accuracies of
88.87% and 82.12% on the PARA and IAE datasets, respectively, using ResNet50 as the backbone. Furthermore, the Emotion Aware Map (EAM) visualization highlights the critical regions of an image, making EAMB-Net more interpretable than its competitors.
## 📃 Dependencies
- pytorch
- torchvision
- tqdm

## 📁 Train
We provide codes that support AVA and PARA datasets.
- Please download the training databases.
- Please download the pretrained emotion-aware model, and then load it in 'EAMBNet.py'
- Please run 'main.py'.

## ⏬ Download
- You can download the AVA database at 
- You can download the PARA database at 
- You can download the pretrained emotion-aware model at [Baidu Cloud](). (password: ) 

## ✨ Statement
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact 1010075746@qq.com or shaofeng@nbu.edu.cn

## 🔍 Citation
If our datasets and criteria are helpful, please consider citing the following papers.
H. Chen et al., "Image Aesthetics Assessment with Emotion-Aware Multi-Branch Network" IEEE TIM, 2024

