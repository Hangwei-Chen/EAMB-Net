# EAMB-Net
The code is coming
This is the source dataset and code for the IEEE TIM paper "Image Aesthetics Assessment with Emotion-Aware Multi-Branch Network".

##  📋 Table of content
1. 📎  [Paper Link](#-paper-link)
2. 💡 [Abstract](#-abstract)
3. 📃 [Requirement](#-requirement)
4. 📁 [AST-IQAD](#-AST-IQAD)
5. 🍎 [SRQE](#-SRQE)
6. ✨ [Statement](#-statement)
7. 💎 [Acknowledgement](#-acknowledgement)
8. 🔍 [Citation](#-citation)
## 📎 Paper Link
- **Article title**: Image Aesthetics Assessment with Emotion-Aware Multi-Branch Network
- **Authors**: Hangwei chen, Feng Shao, Baoyang Mu, Qiuping Jinag
- **Institution**: Faculty of Electrical Engineering and Computer Science, Ningbo University
## 💡 Abstract
The aesthetic and appreciation of an image is the innate human perceptual ability. Emotion, as one of the most basic human perceptions, has been found to have a close relationship with aesthetics. However, explicitly incorporating the learned emotion cues into the image aesthetics assessment (IAA) model remains challenging. Additionally, humans consider both fine-grained details and holistic context information in aesthetic assessments. Therefore, the utilization of emotional information to enhance and modulate the representation of aesthetic features in context and detail is crucial for IAA. With this motivation, we propose a new IAA method named emotion-aware multi-branch network (EAMB-Net). Specifically, we first design two branches to extract aesthetic features related to detail and context. Then, an emotion branch is proposed to reveal the important emotion regions by generating the emotion-aware map (EAM). Finally, the EAM is further employed to infuse emotional knowledge into the aesthetic features and enhance the feature representation, producing the final aesthetic prediction. Experimental results validate that the proposed EAMB-Net can achieve superior performance in score regression, binary classification, and score distribution tasks, obtaining the classification accuracies of
88.87% and 82.12% on the PARA and IAE datasets, respectively, using ResNet50 as the backbone. Furthermore, the EmotionAware Map (EAM) visualization highlights the critical regions of an image, making EAMB-Net more interpretable than its competitors.
## 📃 Requirement
- **Pytorch**
- 
## 📁 AST-IQAD
You can download the AST-IQAD database at [Baidu Cloud](https://pan.baidu.com/s/1imaLNEeh9YmZkCNtSgzrXw). (password: j71y) 

## 🍎 SRQE
1. Please run 'demo.m'.
2. You can obtain three quality scores (i.e., CP, SR, OV). Higher scores mean better quality.

## 💎 Acknowledgement
Our code is borrowed parts from [DISTS](https://github.com/dingkeyan93/DISTS) and [PCRL](https://web.xidian.edu.cn/ldli/paper.html). Thanks to them!

## ✨ Statement
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact 1010075746@qq.com or shaofeng@nbu.edu.cn

## 🔍 Citation
If our datasets and criteria are helpful, please consider citing the following papers.
H. Chen et al., "Quality Evaluation of Arbitrary Style Transfer: Subjective Study and Objective Metric," IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2022.3231041.

