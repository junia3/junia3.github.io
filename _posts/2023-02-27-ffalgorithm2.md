---
title: 딥러닝의 체계를 바꾼다! The Forward-Forward Algorithm 논문 리뷰 (2)
layout: post
description: Forward-forward algorithm
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/222030729-fc5d7460-8d26-4d6a-a1a8-fbc71131c9f1.gif
category: paper review
tags:
- Deep learning
- FF algorithm
- Methodology
- Generative model
- Restricted Boltzmann Machine
---
# 들어가며...
이번 포스트는 바로 이전에 작성된 글인 <U>FF(Forward-forward algorithm)</U> 소개글([참고 링크](https://junia3.github.io/blog/ffalgorithm))에 이어 Hinton이라는 저자가 FF 알고리즘의 학습을 contrastive learning과 연관지어 설명한 부분을 자세히 다뤄보고자 작성하게 되었다.

# Hinton의 RBM(Restricted Boltzmann Machine)에 대하여
머신러닝을 공부했던 사람이라면 모를 수 없는 Andrew Ng이라는 사람이 Hinton과 [인터뷰](https://www.youtube.com/watch?v=-eyhCTvrEtE&list=PLfsVAYSMwsksjfpy8P2t_I52mugGeA5gR&ab_channel=PreserveKnowledge)하면서 인생에 있어 가장 큰 업적을 고르라했을 때, Hinton은 [restricted boltzmann machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)을 손꼽았다.
RBM은 backpropagation과 더불어 딥러닝 연구의 학습 방법론 중 하나인데, 놀랍게도 GAN, Diffusion model처럼 RBM 또한 generative(생성) 모델의 한 축이라고 볼 수 있다.
사실 Hinton은 RBM을 처음으로 제시하지는 않았고 그의 [저서 중 하나](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)에서 학습법에 대해 논하였으며, 다양한 application으로 개념을 올린 사람이라고 할 수 있다.

### Generative model
생성 모델에 대해서 간단하게 설명하자면 기존의 understanding based neural network 구조인 ANN, DNN 그리고 CNN같은 disciminative model과는 다르게 '확률 밀도 함수(<U>Probablistic Density Function</U>)'를 모델링하고자 하는 것이다.
<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222032590-7803a29d-b91c-47af-8799-547927b6ef3d.png" width="800">
</p>
뒤에서 보다 자세히 설명하겠지만 RBM을 이용하는 sampling 방식은 deep neural network의 초기화와 관련이 있으며, <U>특정 distribution을 가지는</U> 데이터가 가장 효과적으로 생성될 수 있는 <U>latent factor</U>를 찾는 과정이 된다.

확률 밀도 함수에 대해 알고자 하는 것은 복잡한 task이기 때문에, 이를 위해서 다음과 같이 예시를 들어보도록 하자. 예를 들어 '<U>사람의 얼굴</U>'을 그럴듯하게 생성하고자 한다고 생각해보자. 얼굴을 구성하기 위해서는 여러 요소들이 필요한데, 이들을 각각 'feature'로 생각하고, feature에는 <U>여러 가지 가능한 변수 상태</U>(state)가 존재한다. 뒤에서 energy based approach로 식을 전개하는 과정에서 <U>변수의 상태</U>가 곧 해당 <U>변수가 차지하는 확률 공간에서의 입지</U>(probability)를 의미하기 때문에 잘 기억해두면 좋다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222034574-6220e53c-6b1d-4fe6-aed6-f2be571f7082.png" width="800">
</p>

즉 어떠한 이미지를 구성하는 각 요소들을 가능한 state의 집합으로 생각하고, 집합의 각 요소들이 가지는 확률 분포를 모델링할 수 있다는 것이다. 꼭 얼굴 이미지가 아니더라도 <U>특정 데이터셋에 대해</U> 이러한 방식으로 <U>세부 요소들</U>을 machine이 모델링할 수 있다면, 각 요소들의 조합을 통해 샘플을 그럴듯하게 만들 수 있다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222035138-704f11a3-cd5e-4b5b-a37e-6cf596b8e29f.png" width="300">
</p>

예컨데 StyleGAN의 경우에는 Synthesizer가 $\mathcal{W}$ space의 latent vector를 style vector로 affine transform하여 coarse detail부터 fine detail까지 그려내는 모습을 볼 수 있는데, 이러한 GAN 구조도 결국 <U>각 hidden layer</U>에서 구현할 수 있는 <U>얼굴 특징에 대한</U> 확률 밀도 함수를 학습한 것과 같다.

### Boltzmann machine
앞서 설명한 내용을 기반으로 기계학습에 pdf를 피팅하기 위해서는 각 state에 대해 <U>computational 구조를 모델링</U>해야하는데, 바로 여기서 가져올 수 있는 모델링이 <U>볼츠만 머신</U>이다. 볼츠만 머신은 생각보다 굉장히 간단한 개념이다. Visible(시각적으로 보이는 부분) part에서 샘플링을 하는 방법이 복잡하기 때문에, hidden(은닉층 부분) part에서 <U>implicit한 요소들에 대한 state를 정의</U>하고자 하는 것이다. 즉 <U>'눈에 보이진 않지만 존재하는 무언가'</U>를 내재하는 그래프 구조를 만들게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222036056-4895ec47-6132-4ed0-b504-369b568cbb85.png" width="300">
</p>

위의 그래프 구조에서 각 노드(노란색/초록색 부분)이 각 state라고 생각하면 된다. <U>노란색</U>을 우리에게는 좀 더 <U>가까운 level에서의 feature</U>(얼굴 이미지에서의 눈, 코, 입), <U>초록색</U>을 우리가 알 수는 없지만, 각 feature node들의 connection에 의한 activation으로 표현되는 은닉 요소들이라고 보면 된다. 하지만 위의 구조를 토대로 계산하게 되면 <U>은닉 요소(hidden state)</U>와 <U>가시 요소(visible state)</U> 간의 관계 이외에도 같은 레이어에서의 요소 간의 관계 또한 정의해야한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222037181-ef2359c5-cc9b-4522-b732-cf8895907538.png" width="800">
</p>

그렇기 때문에 위와 같이 <U>같은 레이어 상에서의 관계를 없앤</U> 모델링을 통해 constraints를 주어, 확률 연산이 보다 간단할 수 있게 구성한 것이 바로 <U>'Restricted'</U> 볼츠만 머신이다. 기본 볼츠만 머신은 각 레이어의 학습을 조건부로 표현할 수 없기 때문에 $p(h,~v)$의 joint를 연산해야하지만, 제한된 볼츠만 머신은 각 레이어의 학습을 조건부인 $p(h \vert v),~p(v \vert h)$로서 정의가 가능하다.

단순히 각 레이어에서의 의존성을 없애게 되면 굉장히 흥미로운 일이 일어나는데, 볼츠만 머신이 우리가 흔히 알고 있는 <U>feed forward neural network 구조처럼</U> 변하게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222039046-50d50184-5bbd-4e8f-9723-150158598807.png" width="400">
</p>

이렇게 단순화된 RBM을 학습하는 방법은 Hinton이 제시했는데, 먼저 간단하게 설명하자면 forward propagation을 통해 visible state로부터 hidden state를 결정하게 되고, 다시 hidden state로부터 visible state 상태를 결정하는 loop를 구성하게 된다.

### RBM의 수학적 모델
RBM은 확률 분포를 학습하기 위한 state의 모델링 방식으로, layer 의존성만 남겨 결국 <U>neural network와 같은 학습이 가능</U>한 형태라고 앞서 설명했다. RBM이 실제로 확률 분포를 학습하는 과정에서의 <U>goodness</U>(잘 학습했음)을 <U>수학적으로 표현하는 과정</U>과 그 <U>구조</U>에 대해서 살펴보도록 하자.

RBM의 구조는 <U>visible unit</U>들로 구성된 <U>visible layer</U>, 그리고 각 layer를 연결하는 weight matrix로 구성된다(노드를 <U>연결하는 선</U>이 각각 weight가 있다고 생각하면 된다).

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222041275-a983201c-8f6f-43af-9311-b6c90ab25756.png" width="800">
</p>

수학적 모델을 정의하는 과정에서 <U>사용될 notation과 구조</U>는 위와 같다. State인 $v, h, b, c$는 모두 벡터가 되며, 이들 간의 관계를 정의하는 $W$는 matrix가 된다. 예를 들어 visible layer의 unit의 갯수가 $d$이고 hidden layer의 unit의 갯수가 $n$이라면,

\[
    \begin{aligned}
        v,~b \in & \mathbb{R}^{d \times 1} \newline
        h,~c \in & \mathbb{R}^{n \times 1} \newline
        W \in & \mathbb{R}^{n \times d}
    \end{aligned}    
\]

이처럼 표현할 수 있다.

### Energy based learning
RBM의 학습에 대한 goodness 또한 cost function으로 정의할 수 있는데, 이때 'Energy'라는 개념을 사용하게 된다. 앞서 각 노드가 의미하는 것은 확률 밀도 함수의 한 축을 담당하는 state라고 했는데, 이때 <U>특정 상태</U> $x$에 대한 에너지 $E(x)$를 곧 해당 상태(state)에 대응하는 값으로 생각해볼 수 있다. 물리학 개념을 생각해보면 에너지가 <U>낮을수록 안정적</U>이기 때문에, 모든 가능한 state $x$에 대한 에너지 분포를 해당 state가 존재할 확률로 normalizing할 수 있게 된다. 바로 다음과 같이 말이다.

\[
    \begin{aligned}
        p(x) =& \frac{\exp(-E(x))}{Z}    \newline
        \text{where } Z =& \sum_x \exp(-E(x))
    \end{aligned}
\]

바로 여기서 왜 해당 모델링이 '볼츠만 머신'이라 불리는지 이해할 수 있다. 통계 역학이나 수학에서 <U>Boltzmann distribution</U>은 시스템이 특정 state의 에너지와 온도의 함수로 <U>존재할 확률을 제공</U>하는 probability distribution function이다. 실제로 볼츠만 머신이 볼츠만 분포와 같이 temperature에 대한 정의를 주지는 않지만, 각 노드가 가지는 position을 해당 상태의 에너지로 정의하고, 이를 정규화한 <U>볼츠만 분포 형태로 근사하고 싶은 것</U>이다.

RBM에서는 visible unit인 $v$와 hidden unit인 $h$의 각 unit state에 따라 에너지를 결정할 수 있는데, <U>hidden unit</U>에 대한 energy는 <U>관찰할 수 없기 때문</U>에 <U>visible unit</U>에 대한 <U>확률 분포를 결정</U>할 수 있다.

\[
    \begin{aligned}
        p(v) =& \frac{\exp(-E(v,~h))}{Z}    \newline
        \text{where } Z =& \sum_v \sum_h \exp(-E(v,~h))
    \end{aligned}
\]

그리고 다시 hidden unit에 의해 복잡해진 energy function을 free energy $F(\cdot)$를 통해 다음과 같이 단순화할 수 있다.

\[
    \begin{aligned}
        p(v) =& \frac{\exp(-F(v))}{Z^\prime}    \newline
        \text{where } F(v) =& -\log \sum_h \exp(-E(v,~h)) \newline
        \text{and } Z^\prime =& \sum_v \exp(-F(v))
    \end{aligned}
\]

Free energy를 해석하면 <U>'모든 hidden state'</U>에 대한 unnormalized negative log likelihood의 총합이고, 이렇게 정의를 할 수 있는 이유는 RBM에서 <U>hidden state 서로에 대한 의존성을 배제</U>했기 때문이다. RBM에서 Energy는 다음과 같이 정의된다.

\[
    E(v,~h) = -b^\top v - c^\top h - h^\top Wv    
\]

Energy 식은 총 세 부분으로 구성된다. Visible layer의 state vector에 대한 biased term($-b^\top v$) 그리고 hidden layer의 state vector에 대한 biased term($-c^\top h$), 마지막으로 두 state간의 weight 관계($-h^\top Wv$)이다. Bias term인 $b$ 그리고 $c$는 <U>각 layer 전반의 특성을 반영</U>한 값이 되고, weight는 bias term이 보지 못하는 <U>레이어 사이의 관계를 반영</U>한 값이 된다고 생각하면 된다.

... 작성중