---
title: GAN을 활용한 이미지 조작(Image manipulation)
layout: post
description: Image translation/manipulation
post-image: https://res.cloudinary.com/devpost/image/fetch/s--C6uTFOry--/c_limit,f_auto,fl_lossy,q_auto:eco,w_900/https://github.com/mnicnc404/CartoonGan-tensorflow/raw/master/images/cover.gif
category: paper review
use_math: true
tags:
- AI
- deep learning
- generative model
- adversarial training
---

Generative model인 GAN은 여러 방면에서 활용될 수 있다.   
- 대표적인 Image synthesis(합성)의 경우 Texture synthesis([PSGAN](https://arxiv.org/abs/1909.06956), [TextureGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf), [Texture Mixer](https://arxiv.org/abs/1901.03447))이 있으며,   
- Image super resolution(화질 높이는 것)([ProGAN](https://arxiv.org/abs/1710.10196), [Progressive face super-resolution](https://arxiv.org/abs/1908.08239), [BigGANs](https://arxiv.org/abs/1809.11096), [StyleGAN](https://arxiv.org/abs/1812.04948))이 있다.
- Image impainting이라는 task는 미완성된 그림이나 사진을 완성하는 작업으로, [Deepfillv1](https://arxiv.org/abs/1412.7062), [ExGANs](https://arxiv.org/abs/2009.08454), [Deepfillv2](https://arxiv.org/abs/1806.03589), [Edgeconnet](https://arxiv.org/abs/1901.00212), [PEN-Net](https://arxiv.org/abs/1904.07475)이 있다.
- Face image synthesis는 얼굴 이미지 합성과 관련된 task로, [ELEGANT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Taihong_Xiao_ELEGANT_Exchanging_Latent_ECCV_2018_paper.pdf), [STGAN](https://arxiv.org/abs/1904.09709), [SCGAN](https://arxiv.org/abs/2011.11377), [Example guided image synthesis](https://arxiv.org/abs/1911.12362), [SGGAN](https://ieeexplore.ieee.org/document/8756542), [MaskGAN](https://arxiv.org/abs/1907.11922)
- Human image synthesis로는 사람의 포즈나 전체적인 윤곽을 생성하는 task가 된다. [Text guided](https://arxiv.org/abs/1904.05118), [Progressive pose attension](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Progressive_Pose_Attention_Transfer_for_Person_Image_Generation_CVPR_2019_paper.pdf), [Coordinate-based](https://openaccess.thecvf.com/content_CVPR_2019/papers/Grigorev_Coordinate-Based_Texture_Inpainting_for_Pose-Guided_Human_Image_Generation_CVPR_2019_paper.pdf), [Semantic parsing](https://arxiv.org/abs/1904.03379)

하지만 오늘 살펴볼 내용은 이것과는 다르게 이미지를 바꾸는 작업, 즉 image manipulation과 관련된 것들을 볼 예정이다. 오늘 게시글과 관련된 내용들을 언급해보자면
1. Image to image translation([CycleGAN](https://arxiv.org/abs/1703.10593), [MUNIT](https://arxiv.org/abs/1804.04732), [DRIT](https://arxiv.org/abs/1808.00948), [TransGaGa](https://arxiv.org/abs/1904.09571), [RelGAN](https://openreview.net/pdf?id=rJedV3R5tm))
2. Image editing([SC-FEGAN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jo_SC-FEGAN_Face_Editing_Generative_Adversarial_Network_With_Users_Sketch_and_ICCV_2019_paper.pdf), [FE-GAN](https://ieeexplore.ieee.org/document/9055004), [Mask-guided](https://arxiv.org/abs/1905.10346), [FaceShapeGene](https://arxiv.org/abs/1905.01920))
3. Cartoon generation([CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf), [PI-REC](https://arxiv.org/abs/1903.10146), [Internal Representation Collaging](https://arxiv.org/abs/1811.10153), [U-GAT-IT](https://arxiv.org/abs/1907.10830), [Landmark Assisted CycleGAN](https://arxiv.org/abs/1907.01424))

---

# Image translation
위에서 많은 task를 언급했던 것은 GAN으로 이만큼이나 많이 할 수 있다는 걸 보여줄라고 한 것이고, 사실 이 게시글의 주 목적은 단순히 image manipulation과 관련된 초기 논문 아이디어에서 insight를 얻어보기 위함이다.   
Image-to-image translation이라 함은 input image로부터 output 이미지를 생성하는 task가 되며, 이때 output과 input은 서로 어떠한 관계에 놓이게 된다.   
이를 테면 computer vision이나 machine learning task에서 주로 나오는 semantic labeling이나 boundary detection이 될 수도 있고,
<p align="center">
    <img src="imagetoimage/001.png" width="400"/>
    <img src="imagetoimage/002.png" width="400"/>
</p>
Computer graphics나 computational photography에서 다루는 image colorization, super-resolution이 될 수도 있다.
<p align="center">
    <img src="imagetoimage/003.png" width="400"/>
    <img src="imagetoimage/004.png" width="400"/>
</p>

즉, 해당 task의 supervision은 source로부터 target을 만드는 과정이 되며, generator $G$는 source domain $S$의 이미지를 사용하여 target image $T$를 만들게끔 학습된다.   
어떠한 문제가 되던, 위와 같은 task는 다음과 같은 objective로 귀결된다.
- Objective function $\mathcal{L}$ 설정
- Training data $(x, y)$ 설정
- Network $G$ 학습하기
- Image translation $G(S) = T$ 정의하기

\[
    \arg \min_{\mathcal{F}} \mathbb{E}_{x,y}(\mathcal{L}(G(x), y))    
\]

그러나 기존의 GAN 방식을 바로 적용하기에는 문제가 있는데, 이는 바로 <U>image generation의 mode(modality)를 제어할 수 없다는 것</U>이다. 즉 우리는 아무런 이미지만 만들면 되는 게 아니라 input으로 넣은 이미지의 translation 버전을 원하는데, 이걸 generator가 인지할 수 없다는 것이 첫번째 문제다. 다음은 generator로 생성한 이미지의 low-resolution 문제가 있다.   
이러한 task를 다룬 총 세 개의 대표적인 paper가 바로 pix2pix, cycleGAN 그리고 pix2pixHD이다. 그 중 가장 유명한 논문인 pix2pix와 cycleGAN에 대해서 먼저 살펴보도록 하자.

   
      
      
... 작성 중 ㅠㅠ