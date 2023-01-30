---
title: About CLIP based image manipulation
layout: post
description: Image manipulation, VL contrastive
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/215319341-b7c913a7-26c9-4bc4-bab9-a19281714eeb.gif
category: paper review
tags:
- Multimodal
- VL
- StyleGAN
- Image manipulation
- Text(prompt) guidance
---

# 들어가며
[StyleGAN](https://arxiv.org/abs/1812.04948)의 등장은 사실상 생성 모델을 style based approach로 해석한 첫번째 논문이라고 볼 수 있다. StyleGAN 논문을 이 글에서 자세히 설명하지는 않겠지만 가볍게 요약하자면, constant vector로 표현된 하나의 feature map 도화지가 있고($4 \times 4 \times 512$), 이 도화지에 latent vector로부터 추출된 style 정보들(평균값인 $\mu$와 표준편차인 $\sigma$)를 affine layer를 통해 얻어내어 이전의 feature map을 modulation하면서 점차 사이즈를 키워나가는 구조다. 점차 feature map의 spatial resolution을 키운다는 점에서 타겟팅이 된 논문 구조인 [PGGAN](https://arxiv.org/abs/1710.10196)도 같이 읽어보면 좋다.

---

# Image manipulation

아무톤 styleGAN의 디테일한 방법론에 대해서 논하고자 하는 것은 아니고, styleGAN이 가져온 이후 연구들의 동향에 대해서 살펴볼 수 있다. 무작위로 뽑아낸 latent vector $z$ 혹은 $w$로부터 '<U>style 정보</U>'를 얻어낼 수 있다는 것은 반대로 생각해서 특정 이미지를 주었을 때, 해당 이미지를 만들어낼 수 있는 latent vector $z$ 혹은 $w$를 추출할 수 있다는 말과 동일하다.   
이러한 GAN inversion 개념은 image manipulation에 큰 동향을 불러왔으며, 특히나 styleGAN의 경우 high-resolution image synthesis가 가능케 한 논문이었기 때문에 <U>고퀄리티의 이미지 조작이 가능하다는 점</U>이 main contribution이 되었다. 이에 여러 이미지 조작 논문들이 나왔으며, 해당 내용에 대해 궁금한 사람들은 본인이 작성한 [image manipulation 포스트](https://junia3.github.io/blog/imagemanipulate)를 참고하면 좀 더 좋을 것 같다.   
StyleCLIP도 결론부터 말하자면 사전 학습된 StyleGAN을 활용한 Image manipulation이라는 task에 대해서 다룬 내용이고, 이 논문에서는 기존 방식과는 다르게 VL contrastive 논문인 CLIP을 활용한 <U>text guidance</U>가 보다 image manipulation에 조작 편의성을 가져다줄 수 있다는 점에 집중하였다.

---

# 기존 방법들의 문제점은?

앞서 언급하기도 했지만 <U>image manipulation task</U>에 대한 연구 동향은 대부분 사전 학습된 styleGAN latent space에서 유의미한 semantical information을 찾고, 이를 조작하는 방식으로 진행된다. 그러나 직접 latent space를 searching하는 과정 자체는 구체적으로 이미지 상에서 어떤 스타일을 변화시킬지도 모르고, 무엇보다 사용자가 원하는 이미지 조작이 이루어지기 힘들다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215320268-1ddc118f-2e02-4fe1-be7a-919196a3d324.png" width="900"/>
</p>

예를 들어 위와 같은 이미지를 보면, 평범한 표정의 여자 이미지를 만들어내는 latent vector $w_1$이 있고, 이와는 다르게 놀라는 표정의 여자 이미지를 만들어내는 latent vector $w_2$가 있다고 해보자. 단순히 latent space를 searching하는 과정에서는 '놀라는 표정'을 만들어낼 수 있는 latent manipulation 방향을 알 수 없기 때문에 random하게 찾는 과정을 거칠 수 밖에 없다. Latent vector는 $512$ 크기의 차원 수를 가지기 때문에 supervision이 없다면 인간이 직접 찾아야하는 번거로움을 피할 수 없다. 이러한 방법을 사용한 것이 [GANspace](https://arxiv.org/pdf/2004.02546.pdf), [Semantic face editing](https://arxiv.org/pdf/1907.10786.pdf) 그리고 [StyleSpace 분석](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_StyleSpace_Analysis_Disentangled_Controls_for_StyleGAN_Image_Generation_CVPR_2021_paper.pdf) 논문들에 해당된다. 이러한 문제들을 해결하기 위한 방식이 attribute에 대한 classification을 guidance로 삼는 [InterfaceGAN](https://arxiv.org/pdf/2005.09635.pdf)이나 [StyleFlow](https://arxiv.org/pdf/2005.09635.pdf)가 제안되었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/215321251-44e2194d-1ae1-4fc8-857e-ac61137e92e4.png" width="900"/>
</p>

만약 latent space에서 특정 element를 바꿨을 때의 attribute 변화가 '얼굴 표정'임을 원한다면 해당 attribute의 변화를 최대화하는 쪽으로 latent manipulation을 진행하게 되는 것이다. 이외에도 parametric model인 3DMM을 활용하여 3D face mesh에 consistent한 sampling을 진행하는 [StyleRig](https://arxiv.org/pdf/2004.00121.pdf) 논문 등등이 소개되었다.   
어찌되었든 기존 방식들은 정해진 semantic direction을 찾아가는 과정을 바꾸지는 못했고, 이는 사용자의 image manipulation 방식에 제한을 둘 수밖에 없었다(latent space를 searching하거나, classifier에 의한 guidance). 기존 방식에서 벗어나 latent space에 mapping되지 않은 direction에 대해서 searching하는 과정을 고려하면(StyleRig), <U>manual effort</U>와 충분한 갯수의 <U>annotated data</U>가 필요한 것을 알 수 있다.

---

# CLIP을 활용한 image manipulation
따라서 저자들은 방대한 text-image web dataset으로 prompt 기반 image representation을 학습한 VL contrastive model의 성능에 집중하였고, 해당 네트워크가 <U>prompt based zero-shot image classification</U>에서도 성능을 입증했던 바와 같이 마찬가지로 image manipulation 과정에서도 어느 정도 자유도를 보장할 수 있다고 생각하였다. CLIP 네트워크와 기존 framework인 styleGAN을 혼용하는 방식은 여러 가지가 있을 수 있지만, 저자들은 이 논문에서 총 세 가지의 방법들을 언급하였다.

1. 각 이미지 별로 text-guided latent optimization을 진행하는 방법. 단순히 CLIP model을 loss network(latent vector를 trainable parameter로 gradient를 보내고, clip encoder의 결과와 text encoder의 결과 사이의 similarity를 loss로 주게 되면 latent 최적화가 진행되는 방식)
2. Latent residual mapper를 학습하는 방법. 이 과정에서는 latent vector를 직접 최적화하는 것이 아니라 latent mapper를 학습하여, $W$ space의 특정 latenr를 CLIP representation에 맞는 또다른 latent vector로 mapping한다.
3. 이미지에 상관 없이 text prompt에 맞는 style space를 학습하는 방법. 이 방법은 styleGAN이 가지고 있는 $\mathcal{W}$ space의 disentanglement 특징을 그대로 유지하면서 style mapping을 하고자 하는 것이 주된 목적이다.

...작성중