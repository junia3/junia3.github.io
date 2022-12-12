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

---

# pix2pix: Image-to-Image Translation with Conditional Adversarial Nets
<p align="center">
    <img src="imagetoimage/005.png" width="600"/>
</p>
pix2pix는 대표적인 image to image translation을 GAN으로 해결한 연구이다. 가장 유명한 figure인 sketch to real image에 대한 framework는 위와 같다.   
<p align="center">
    <img src="imagetoimage/006.png" width="600"/>
</p>
만약, generator가 단순히 sketch image를 가지고 real image를 만들어내는 task에 대해서 생각해보면, 위의 그림과 같이 '그럴듯한 가방'을 만들어내는 것도 가능하지만 그럴 경우 실제 sketch와의 correspondence 문제까지 고려해야한다. 즉, 위쪽 row에 대해서는 sketch 부분에 잘 맞게끔 이미지가 생성되지만, 아래쪽의 row에서는 sketch는 거의 무시한 채 이미지를 생성해낸다.   
이러한 문제를 기존 GAN loss에서는 고려할 수 없었으며(아래쪽 식을 참고),
\[
    \begin{aligned}
        &\min_G \max_D V(D,G) \newline
        V(D,G) =& \mathbb{E_{x \sim p_{data}(x)}}(\log D(x)) + \mathbb{E_{z \sim p_z(z)}}(\log (1-D(G(z))))
    \end{aligned}    
\]
이는 generator가 만든 이미지에 대해 loss를 적용할 때 단순히 real distribution에 의한 결과인지 cross-entropy로 구분했기 때문이다. 물론, 이 식을 그대로 적용하지는 않고 input으로 sketch image를 주기 때문에 데이터셋 $(x, y)$를 적용한 GAN loss를 고려해보면,

\[
    \begin{aligned}
        &\min_G \max_D V(D,G) \newline
        V(D,G) =& \mathbb{E_{x \sim p_{data}(x)}}(\log D(y)) + \mathbb{E_{z \sim p_z(z)}}(\log (1-D(G(x, z))))
    \end{aligned}    
\]

Generator에 input $x$가 latent vector $z$와 함께 주어지는 구조인 걸 확인할 수 있다. 그러나 이러한 식은 실질적으로 discriminator가 generator에게 줄 수 있는 학습 정보는 "진짜같은" 이미지인지에 대한 loss 뿐이므로 correspondence를 해결할 수 없다는 문제가 생긴다.   
그래서 제시된 objective function이 conditional GAN의 방법을 이용한 loss이며, 여기에 추가적으로 MAE(Mean absolute error)를 생성된 이미지와 정답(GT) 이미지 사이에 줌으로써 low resolution result 문제와 correspondence 문제 모두 해결하려 했다.

\[
    \begin{aligned}
        &\arg \min_G \max_D \mathcal{L} (G,D) + \lambda \mathcal{L}(G) \newline
        \mathcal{L}(G, D) =& \mathbb{E}(\log D(x, y))+\mathbb{E}(\log(1-D(x, G(x, z)))) \newline
        \mathcal{L}(G) =& \mathbb{E}(\parallel y-G(x, z) \parallel_1)
    \end{aligned}  
\]

위에 표현된 식에서 $\mathcal{L}(G, D)$는 conditional GAN loss에 해당되며, 앞서 설명했던 것과 같이 discriminator에 input 정보를 함께 줌으로써 생성된 이미지가 입력된 이미지에 대한 조건을 가지게끔 해준다. 그러나 해당 loss는 content가 유지된다는 보장을 줄 수 없기 때문에 여기에 추가적으로 $\mathcal{L}(G)$로 표현된 식을 통해, 원본 이미지와 유사하게 생성되게끔 만들어준다. $\lambda = 100$ 정도로 크게 생각해주면 된다.
<p align="center">
    <img src="imagetoimage/007.png" width="600"/>
</p>

생성자 구조는 이와 같이 [U-Net](https://arxiv.org/abs/1505.04597) 형태를 이용하였다. 또한 이 논문에서 특별한 점은 $x$에 추가적으로 latent vector $z$를 넣어주지 않고, decoder part의 dropout으로 해당 stochastic한 부분을 충당할 수 있다고 한다. 그래서 사실상 loss 식에서 표현된 $z$를 따로 넣어주지는 않는다. 추가로 넣어주더라도 별로 효과적이진 않다고 판단했다. 즉, 넣어주어도 결국 $z$에는 아무런 영향을 못받는다고 했다.

<p align="center">
    <img src="imagetoimage/008.png" width="600"/>
</p>

각 loss term에 의한 효과를 보여주는 그림이다. L1 loss를 쓰지 않았을 때는 GT의 전체적인 틀을 잃어버리는 문제가 발생하고, cGAN loss를 쓰지 않았을 때는 blurry한 결과가 나오는 것을 확인할 수 있다. Discriminator 구조는 appendix에 따로 나와있는데, 간단하게만 설명하면 모든 ReLU는 LeakyReLU(기울기 0.2)를 사용하였고 [DCGAN](https://arxiv.org/abs/1511.06434)에서와 같이 첫번째 layer에서는 BatchNorm을 사용하지 않았다. 이러한 pix2pix는 두 개의 paired dataset만 있다면 한쪽을 source, 다른 쪽을 target으로 삼아서 다양한 image to image translation task에 적용될 수 있다는 장점이 있다.   
그러나 여기서 생길 수 있는 문제는, 과연 input-output 간에 paired dataset이 없다면 어떻게 될까이다. 이를테면 sketch dataset에 그에 상응하는 사물 이미지가 있어야 sketch to image generation 학습이 가능하고, scene에 대한 depth map이 존재해야 depth estimation 생성이 가능하기 때문이다. 즉 한계점은 pair를 모으기 힘들고, 몇몇 task에 대해서는 아예 불가능할 수도 있다는 것이다. 바로 이러한 관점에서 제시된 것이 cycleGAN이다.

... 계속 작성중