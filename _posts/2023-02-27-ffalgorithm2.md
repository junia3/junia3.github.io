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

---

# Bernoulli RBM

앞서 설명했던 <U>얼굴 생성과 관련된 state</U>는 각각 여러 상태를 가질 수 있지만, Bernoulli RBM의 경우에는 visible/hidden unit 각각 $0$ 혹은 $1$의 상태만 가지는 경우를 다루게 된다. 앞서 살펴본 free energy $F(\cdot)$로 전개된 식을 살펴보면,

\[
    \begin{aligned}
        F(v) =& -\log \sum_h \exp (-(-b^\top v - c^\top h - h^\top Wv)) \newline
        =& -\log \sum_h \exp(b^\top v + c^\top h + h^\top Wv) \newline
        =& -\log \sum_h \exp(b^\top v) \exp(c^\top h + h^\top Wc) \newline
        =& -\log \left( \exp(b^\top v) \sum_h \exp(c^\top h + h^\top Wc) \right) \newline
        =& -b^\top v -\log \sum_h \exp(c^\top h + h^\top Wc)
    \end{aligned}
\]

위와 같이 정리할 수 있다. 그리고 $h$가 곧 $0$ 혹은 $1$ 이므로(Bernoulli)

\[
    \begin{aligned}
        -b^\top v& - \sum_{i=1}^n \log (\exp (0) + \exp (c_i + W_i v)) \newline
        =& -b^\top v - \sum_i \log (1 + \exp(c_i + W_i v))
    \end{aligned}    
\]

위와 같이 표현할 수 있다. RBM은 input에 대해 의존하는 neural network 구조와는 다르게 visible layer를 이용하여 hidden layer의 state를 생성할 수도 있지만, <U>반대로 hidden layer를 이용하여</U> 다시 <U>visible layer를 생성</U>할 수 있다. Visible layer가 주어졌을 때의 조건부 확률을 energy based로 전개하면 다음과 같다.

\[
    \begin{aligned}
        p(h \vert v) =& \frac{p(h,v)}{p(v)} = \frac{\exp (-E(h, v))/Z}{\sum_h p(h, v)} \newline
        =& \frac{\exp(-E(h, v))/Z}{\sum_h \exp(-E(h, v))/Z} \newline
        =& \frac{e^{b^\top v} e^{c^\top h + h^\top Wv}}{\sum_h e^{b^\top v} e^{c^\top h + h^\top Wv}} \newline
        =& \frac{e^{c^\top h + h^\top Wv}}{\sum_h e^{c^\top h + h^\top Wv}}
    \end{aligned}    
\]

총 $n$개의 hidden unit이 있을때, 이 중에서 하나의 hidden unit이 $1$인 값을 가질 확률은 sigmoid 함수인 $\sigma(\cdot)$으로 표현할 수 있다.

\[
    \begin{aligned}
    p(h_i = 1 \vert v) =& \frac{e^{c_i + W_i v}}{\sum_h e^{c_i h_i + h_i W_i v}} \newline
    =& \frac{e^{c_i + W_i v}}{e^0 + e^{c_i + W_i v}} \newline
    =& \frac{e^{c_i + W_i v}}{1 + e^{c_i + W_i v}} = \frac{1}{1 + \frac{1}{e^{c_i + W_i v}}} \newline
    =& \sigma (c_i + W_i v)
    \end{aligned}    
\]

반대 방향에 대해서도 <U>같은 공식을 적용</U>할 수 있고, 이때 바뀌는 것은 bias와 weight에 곱해지는 input이기 때문에

\[
    p(v_j = 1 \vert h) = \sigma(b_j + W_j^\top h)    
\]

와 같다. RBM 모델의 수학적 모델링은 위의 공식대로 sigmoid 함수를 따르는 조건부 확률이 되며, RBM이 학습하고자 하는 것은 <U>데이터의 확률 분포</U>이다.
만약 RBM의 hidden layer가 적절한 property에 대한 distribution $p(h)$를 제대로 학습했다면, sampling을 통해 획득할 수 있는 $p(v \vert h)$가 원래 데이터인 $p(v)$와 같아야 한다. 마치 Variational autoencoder랑 비슷하긴한데 조금 다른 점은 VAE 구조는 encoding이 목적이고 RBM은 <U>확률 밀도 함수 자체를 레이어에 피팅</U>하고자 하는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222154860-16a4c59c-576d-4477-8e7d-fab9c3700f98.png" width="800">
</p>

이제 실제로 왜 $p(v)$에 대한 negative log likelihood를 최적화하는 것이 <U>contrasitve learning과 관련될 수 있는지</U> 수식으로 증명할 수 있는 이론적 배경이 완성되었다. FF 설명하고자 여기까지 왔다... 

---

# Parametric learning in RBM
RBM의 파라미터를 $\theta$라고 해보자. 여기서 parameter란 weight와 bias가 될 수 있다. 앞서 전개했던 식들을 토대로 negative log likelihood를 구하면 다음과 같다.

\[
    \begin{aligned}
    -\frac{\partial}{\partial \theta} \log p(v) =& -\frac{\partial}{\partial \theta} \log \left( \frac{\exp (-F(v))}{Z} \right)   \newline
    =& -\frac{\partial}{\partial \theta} \left( \log \exp(-F(v)) -\log (Z) \right) \newline
    =& -\frac{\partial}{\partial \theta}(-F(v) - \log (Z)) \newline
    =& \frac{\partial}{\partial \theta} F(v) + \frac{\partial}{\partial \theta} \log (Z) \newline
    =& \frac{\partial}{\partial \theta} F(v) + \frac{\partial}{\partial \theta} \log \left( \sum_{\tilde{v}} \exp (-F(\tilde{v})) \right)
    \end{aligned} 
\]

$\tilde{v}$는 RBM에 의해 생성되어 visible unit에 정의된 state vector를 의미한다. 즉, <U>은닉층으로부터 생성된 샘플</U>을 의미한다.

\[
    \begin{aligned}
        =& \frac{\partial}{\partial \theta} F(v) + \frac{\sum_v \exp (-F(\tilde{v})) \frac{\partial}{\partial \theta} (-F(\tilde{v}))}{\sum_{\tilde{v}} \exp(-F(\tilde{v}))} \newline
        =& \frac{\partial}{\partial \theta} F(v) - \sum_{\tilde{v}} \frac{\exp (-F(\tilde{v}))}{Z} \cdot \frac{\partial}{\partial \theta} (F(\tilde{v})) \newline
        =& \frac{\partial}{\partial \theta} F(v) - \sum_{\tilde{v}} p(\tilde{v}) \frac{\partial}{\partial \theta} F(\tilde{v})
    \end{aligned}    
\]

식을 정리하고 났더니 <U>생성된 샘플</U> $p(\tilde{v})$에 의한 <U>자유 에너지 변화율 평균</U>이 되었다. 확률 기댓값으로 정의되는 구조이기 때문에

\[
    = \frac{\partial}{\partial \theta} F(v) - E_\tilde{v} \left( \frac{\partial F(\tilde{v})}{\partial \theta} \right)    
\]

이처럼 정의할 수 있다. 물론 <U>당연한 이야기지만</U> feasible visible sample $\tilde{v}$에 대한 기댓값 연산이 불가능하므로 <U>샘플의 평균을 통해 근사하는 학습</U>이 진행된다.

\[
    \approx \frac{\partial}{\partial \theta} F(v) - \frac{1}{\vert \mathcal{N} \vert} \sum_{\tilde{v} \in \mathcal{N}} \frac{\partial F(\tilde{v})}{\partial \theta}
\]

앞서 말했던 바와 같이 볼츠만 머신은 각 레이어가 의미하는 것이 특정 representation의 확률 분포가 되고, 따라서 위의 식은 <U>서로 다른 두 분포</U>(하나는 원래 데이터, 다른 하나는 모델링된 데이터)의 간격을 줄이는 것과 같다. 바로 익숙한 KL divergence로 생각해볼 수 있으며, 이는 <U>energy based approach</U>에서 <U>에너지의 변화가 곧 확률 분포의 변화</U>이기 때문이다.
샘플링할 수 있는 $\tilde{v}$의 개수에 따라 loss term은 달라지지만, Hinton 교수는 한 번의 샘플링으로 gradient descent를 사용하더라도 RBM 학습이 가능하다고 밝혔고, loss 식은 <U>다음과 같이 단순화하여 표현</U>할 수 있다.

\[
    \text{loss} = F(v) - F(\tilde{v})
\]

바로 이러한 맥락에서 유도된 <U>볼츠만 머신에서의 contrastive divergence</U>란 real data와 네트워크가 만들어낸 가상의 data 사이의 간격을 줄이는 학습을 의미한다.

\[
    \frac{\partial KL(P_\text{data} \vert\vert P_\text{model})}{\partial w_{ij}} = \left< s_i s_j \right>_\text{data} - \left< s_i s_j \right> \text{model}
\]

---

# Relationship with RBM and FF algorithm 

식에서의 brackets는 <U>레이어 사이의 state</U>가 fluctuation(weight에 따른 변화)하는 것을 표현한다. 앞에서 유도한 식을 일반적으로 표현한 것과 같다. 결국 RBM이 학습하는 것이 네트워크 전체에 대해 error를 propagation하는 구조가 아니라 두 개의 레이어 사이에 <U>real sample과 modeled sample을 유사하게</U> 만드는 것이다. 볼츠만 머신에 대한 아이디어는 저자가 간단하게 다음과 같이 정리해주었다.

1. Learn by minimizing the free energy $F(\cdot) on real data and maximizing the free energy on negative data generated by network$
2. Use the Hopfield energy as the energy function and use repeated stochastic updates to sample global configurations from the Boltzmann distribution defined by the energy function

저자가 추가로 FF 알고리즘에 대해 언급하는 것은 <U>wake</U>는 일종의 <U>bottom-up</U> 구조로, real data를 통해 hidden state에 대한 representation을 축적하고, <U>sleep</U>은 <U>top-down</U> 구조로, 학습된 hidden state로 sampling하는 것과 같다. 그렇기 때문에 RBM 학습 방법이 이전 글에서 살펴보았던 구조랑 동일한 해석으로 이어진다는 것. Hinton 씨는 아마도 <U>볼츠만 머신</U>이 간단한 iterative 구조를 가짐으로써 <U>복잡한 task에 적용되지 못하고</U> backpropagation 알고리즘에 비해 제대로 연구가 진행되지 못한 점을 아쉽게 생각한 듯하다.

# Relationship with GAN

그리고 <U>backpropagation</U> 방법을 통한 generative model 중 유명한 녀석인 <U>GAN</U> 역시도 이와 비슷하다. FF 알고리즘이 각 레이어마다 greedy algorithm을 통해 iterative한 최적화를 진행한다면, GAN은 disciminator가 <U>네트워크로 생성된 데이터</U>가 positive sample인지 negative sample인지 구분하게 된다. 다만 GAN과는 다른 점은 <U>probability 학습에 사용된 layer</U>가 그대로 goodness 판별에 사용되므로, backpropagation이 불필요하다는 것이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/222173754-41942750-2d49-4204-b183-f881b74e5303.png" width="800">
</p>

---

... 작성중