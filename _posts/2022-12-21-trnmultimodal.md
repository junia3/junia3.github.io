---
title: Transformer와 Multimodal에 대하여
layout: post
description: Attention mechanism
use_math: true
post-image: https://media4.giphy.com/media/9AjXOu17dKtiCUvAwg/giphy.gif?cid=ecf05e47kd5zogug0mvc1c0ggtezexjljyqergxk2yzkaou4&rid=giphy.gif&ct=g
category: paper review
tags:
- Transformer
- Attention
- AI
- Deep learning
---

# Convolutional neural network의 발전
기존의 딥러닝에서 사용되던 대부분의 네트워크 구조는 <U>Multilayer perceptron</U>(MLP) 혹은 <U>Convolutional neural network</U>(CNN)이 주를 이루고 있었다. 단순히 modality를 $1 \times N$ 차원 벡터로 늘려서 연산하는 MLP와는 다르게 Convolutional neural network 구조는 image와 같은 modality에서 성능을 입증받았고, 다양한 연구들이 그 뒤를 이었다. Convolutional neural network가 MLP에 비해 가지는 장점은 많았다. 우선 첫번째로 MLP보다 동일한 hidden layer 개수를 가지는 deep neural network를 구성함에 있어 적은 parameter를 학습시켜도 된다는 점과, 학습 가능한 parameter 수는 더 적으면서도 학습 가능한 representation이 <U>modality에 대해</U> 일반화가 잘된다는 점이다.   
이러한 퍼포먼스 덕분에 NLP(Natural language processing)이나 Audio 관련 딥러닝에서도 CNN 구조를 통해 연구를 시작했으며, 이는 'Inductive bias' 덕분이라고 언급할 수 있다. <U>Inductive bias</U>란, 학습하는 모델이 관측하지 못할 모든 modality에 대해서 추정하기 위해 학습이나 추론 과정에서 주어질 수 있는 모든 '가정'의 집합이다. 이게 무슨 의미냐면 예를 들어 딥러닝 네트워크가 '왼쪽을 보는'라는 고양이에 대한 데이터셋이 없이 고양이에 대한 classification을 학습했음에도, 추론 과정에서 '왼쪽을 보는' 고양이 이미지가 주어졌을 때 해당 객체가 고양이라는 사실을 인지하도록 하기 위한 일종의 constraint라고 보면 된다. 물론 이 예시는 사실 약간 부적절한 내용이지만 **inductive bias**라는 용어가 transformer라는 모델에 대한 개념을 파악하기에 필수적이기 때문에 이를 먼저 간단하게 이해하고 넘어가고 싶었다.
<p align="center">
    <img src="transformer/000.png" width="600"/>
</p>
위의 그림을 보면 간단하게 inductive bias에 대해서 설명할 수 있다. Convolution 연산이 진행되는 과정을 보면 필터가 특정 영역(ex. $3 \times 3$)을 기준으로 필터의 paramter와 feature 값을 모두 multiplication한 뒤 더하는 구조가 된다. 따라서 다음과 같은 두 가지의 가정을 해볼 수 있다.

- 이미지에 속한 object의 경우, 해당 object를 나타내는 픽셀의 상대적 위치는 보장된다(localization).
- 이미지에 속한 object가 움직일 경우, 해당 object에 대한 feature output 또한 이미지 상에서 움직인다(translation equivariance).

첫번째 조건의 경우엔 큰 문제가 없으나, 두번째 조건은 큰 문제가 생긴다. 왜냐하면 translation equivariance는 MLP에서도 처리하지 못했던 문제였고, 만약 같은 object의 <U>위치만 달라짐</U>으로써 얻어지는 <U>feature map의 형태에 변화</U>가 생긴다면, 같은 object에 대해 동일한 예측을 취할 수 있다는 게 보장되지 않기 때문이다. 물론 MLP와는 다르게 CNN에서는 물체의 형태가 달라지지는 않고, 단순히 그 상대적인 위치는 유지된 채로 feature map 상에서의 전반적인 localization만 바뀐다는 것이다.   
그러나 CNN의 구조 상에서 max-pooling과 같이 같은 kernel 내의 feature value를 단일한 값으로 축약하는 filtering module, 그리고 softmax를 통한 확률값 계산을 통해 localization이 보장된 feature map에 대해 동일한 예측값을 낼 수 있다는 사실이 확인되었다. 즉 <U>feature extraction</U> 부분은 translation equivariance와 localization을 기반으로 object에 대해 유의미한 feature 형태를 유지하는 과정이 되며, <U>classifier</U> 부분은 translation invariance를 통해 이렇게 추출된 feature map을 기반으로 일관성 있는 prediction을 하게된다.

<p align="center">
    <img src="transformer/001.png" width="600"/>
</p>

바로 이러한 장점 덕분에 convolutional neural network는 더 적은 parameter 수를 가지고도 같은 class에 대한 feature representation 학습에 유리했고, ImageNet에서의 첫 성공 이후 수많은 연구가 진행되었던 것이다. 무엇보다 sequential data를 처리해야하는 NLP, Audio, Video 등등 temporal information을 추출하는데에도 CNN 구조를 많이 사용하기 시작했다.

<p align="center">
    <img src="transformer/002.png" width="600"/>
</p>

그러면서 자연스럽게 sequential data를 처리하기 위한 <U>RNN</U>(Recurrent neural network)이 등장하게 되었으며, LSTM, GRU와 같은 long-term memory 모듈을 활용하여 input과 output에 대한 global contextual meaning을 학습하려는 노력이 시작되었다.

---

# 기계 번역에서의 RNN과 한계점
<p align="center">
    <img src="transformer/003.png" width="600"/>
</p>
예를 들어 '<U>나는 고양이 이름을 에폭이라 짓기로 결정하였다</U>'라는 한국어 문장을 '<U>I decided to name my cat Epoch</U>'라는 영어 문장으로 번역하는 것을 머신러닝/딥러닝으로 해결하려는 기계 번역 task가 있다고 생각해보자. 단순히 이런 task를 RNN의 관점에서 접근하게 되면,   
<p align="center">
    한국어 문장 $\rightarrow$ $E_{\theta}$ $\rightarrow$ 임베딩 $\rightarrow$ $D_{\phi}$ $\rightarrow$ 영어 문장
</p>

위와 같이 표현할 수 있다. 중간에 있는 $E_\theta, D_\phi$는 각각 인코더와 디코더를 의미한다. 이러한 <U>Encoder-Decoder 구조</U>를 가지는 RNN 형태로 대표적인 것이 sequence-to-sequence 모델이다.

<p align="center">
    <img src="transformer/004.png" width="700"/>
</p>

첫번째 가장 큰 문제점은 연산 속도가 된다. 딥러닝이 레이어를 깊게 가져가면서도 연산 속도를 빠르게 할 수 있었던 것은 텐서 연산에 대해 GPU를 통한 병렬 처리가 가능했다는 점이다. 같은 level에서의 feature map value는 동일하기 때문에 convolution 연산을 <U>굳이 순차적으로 진행하지 않고도</U> 모든 연산을 **동시에** 할 수 있기 때문에 구조적 이점을 가져갈 수 있었다. 그러나 RNN의 경우는 그럴 수 없었다. 구조를 보면 알 수 있지만 LSTM(hidden layer로 사용된다고 보면 된다)의 각 계산을 하기 위해서는 이전 LSTM의 결과가 필요하다. 병렬 처리를 통해 LSTM 하나의 연산을 빠르게 한다고 하더라도 문장의 길이 $N$에 대해서는 여전히 <U>bottleneck</U>이 걸리고 있는 것이다. 두번째 문제점은 연산 과정에서 사용되는 context vector의 크기가 고정적이라는 것이다. 이 또한 문장의 길이가 길어질수록 큰 문제가 생기는데, RNN 구조에서 번역 과정에서 참고할 수 있는 feature embedding은 오직 encoder의 final hidden layer의 output이다. 따라서 복잡한 문장이나 번역이 까다로운 경우 문장의 길이가 길어질수록 encoder에서의 문장을 제대로 참고하지 못하는 문제가 발생하였다. 바로 이러한 문제로부터 등장한 것이 attention 메커니즘이고, transformer의 근간이 되는 기술이기도 하다.

---

# Attention mechanism
<p align="center">
    <img src="transformer/005.png" width="500"/>
</p>

앞서 말했듯이 RNN에는 크게 두 가지의 문제점이 있다. 첫번째는 long sentence에 대해 연산 속도가 너무 느리다는 점이고, 두번째는 문장이 길어지면 길어질수록 context vector의 <U>고정적인 길이</U>에 대해 performance 제약을 받는다는 점이다. 이러한 문제를 해결하기 위해서는 RNN을 구성하는 <U>모든 LSTM의 output에 대해</U> reasoning이 필요하게 되었고, 여기서 'attention'이라는 모듈이 제시가 되었다. Attention module은 input에 대해(single input이 될 수도, multiple input이 될 수도 있음) 서로 얼마나 연관도를 가지는지를 weight로 표현한 tensor를 추출한다. 예를 들어 다음과 같은 문장이 제시가 되었다고 생각해보자.

\[
    \text{I decided to name my cat Epoch}
\]

이를 단어 단위로 tokenize한 뒤에,

\[
    \text{(I, decided, to, name, my, cat, Epoch)}
\]

각 token을 모두 embedding 조건에 따라서 value로 mapping한 뒤, 단어 '<U>name</U>'에 대해서 같은 문장에 있는 단어들과의 유사성을 구해보려고 한다. 우선 알 수 있는 사실은 'name'과 가장 관련이 있는 단어는 '<U>Epoch</U>'이며, 그 다음으로 중요한 단어는 '<U>cat</U>'이 될 수 있을 것이다.

\[
    \text{(I, decided, to, name, my, cat, Epoch)} \rightarrow (0, 0, 0, 0.3, 0, 0.1, 0.6)
\]

극단적인 경우로 가정했지만, 아무튼 이처럼 input의 각 embedding 사이의 유사성을 weight로 매핑하고, 이렇게 추출된 <U>weight의 softmax 확률값</U>을 **attention value**로 삼게 되는 것이다. 이렇게 attention을 사용하게 되면 LSTM와 같은 long term 모듈에 의지하지 않고도 문장 전체에 대한 global correlation 혹은 long range interaction을 획득할 수 있다.   
그러나 여전히 이러한 sequence to sequence를 사용해서도 해결할 수 없는 문제가 있는데, 그것은 바로 decoder에 사용될 feature vector를 추출하기 위한 RNN의 '<U>순차적 연산</U>'을 가속화할 방법이다. 여전히 학습 성능과 추론 시간 사이에 해결할 수 없는 trade-off가 남아있게 된다. 이제 드디어 설명하려는 것이 이러한 bottleneck이 해결될 수 없는 RNN 구조에 <U>의존하지 않고</U> 유의미한 기계 번역을 할 수 있다는 내용의 새로운 패러다임을 제시한 <U>transformer</U>가 되겠다.

---

# Attention is all you need!
<p align="center">
    <img src="transformer/006.gif" width="700"/>
</p>
[Transformer](https://arxiv.org/abs/1706.03762)가 바로 이 제목과 함께 <U>거대한 어그로</U>를 끌며 나타났다. 해당 논문의 main idea는, 굳이 RNN과 같은 convolution 구조를 사용해서 global context를 뽑지 않더라도 어차피 attention을 쓰면 즉각적으로 global reasoning이 가능하고, 이러한 attention 연산을 여러번 진행하는 deep neural network 구조가 오히려 기계 번역과 같은 task에 더 적합하지 않겠냐는 것이다. 실제로 기존 RNN에 비해서 연산 속도를 줄이면서도 그 성능을 입증받았고, 이러한 <U>구조적 변화</U>는 NLP 및 여러 연구에 변화를 일으키는 초석이 되었다.
<p align="center">
    <img src="transformer/007.png" width="500"/>
</p>
지금부터 transformer 구조를 구성하는 중요 요소 중 하나인 self-attention, multi-head attention 그리고 positional encoding과 masked attention에 대해서 알아보도록 하자.

## Self Attention
RNN에서는 encoder 구조를 통해 feature vector를 추출하였다. 이 feature vector는 모든 input token에 대한 context를 담고 있게끔 학습된다는 것이 RNN 구조에서의 assumption이었고, 여기에 추가적으로 attention value를 통해 이전의 RNN feature vector까지 고려했던게 기존 방식이다. <U>Attention mechanism</U>만을 사용하는 **transformer**는 이러한 의존성을 없애고, 단순히 attention을 같은 문장에 여러번 적용함으로써 이러한 contextual embedding을 추출한다. Attention에서 사용되는 용어에 대해서 먼저 짚고 넘어가도록 하자.

- Query($Q$) : '질문' 혹은 '물음'이라는 의미를 가지고 있다. 문장 내의 각 토큰이 다른 토큰과 어떤 연관을 가지고 있을지 추론해가는 과정에 있어서 '각 토큰'을 의미한다. 예를 들어 위의 예시에서 봤던 것처럼 단어 '<U>name</U>'에 대해서 같은 문장(I decided to name my cat Epoch)에 있는 단어들과의 유사성을 구해보려고 할 때의 'name'과 같다.

- Key($K$) : Query와 pair로 작용하여, query가 특정 key에 대해 value를 물었을 때의 이를 줄 수 있는 역할을 한다. 예를 들어 위의 예시에서 봤던 것처럼 단어 '<U>name</U>'에 대해서 같은 문장(I decided to name my cat Epoch)에 있는 단어들과의 유사성을 구해보려고 할 때 문장 내에 있는 단어들(I, decided, cat 등등)과 같다.

- Value($V$) : Key에 대응되는 value를 의미한다. Query, key, value에 대한 개념은 단일로 이해하는 것보다는 셋의 <U>연관성</U>을 생각해보는 것이 훨씬 간단하다.

입력으로 주어지는 문장 $S$가 있고 이를 embedding 함수 $\mathcal{E}$를 통해 embedding tensor $X$로 만들었다고 해보자. 이 embedding tensor에 대해 query, key, value는 각각의 weight에 따라 liear projection 된다.

\[
    \begin{aligned}
        Q =& W_1X \newline
        K =& W_2X \newline
        V =& W_3X
    \end{aligned}    
\]

예를 들어 문장을 총 $n$개의 토큰으로 나눈 뒤, 각 토큰을 $e$의 dimension을 가지는 텐서로 치환했다고 하자. $Q,~K,~V$의 dimension $d$에 대해서,

\[
    \begin{aligned}
        X \in & \mathbb{R}^{n \times e} \newline
        K \in & \mathbb{R}^{n \times d},~W_1 \in \mathbb{R}^{e \times d} \newline
        Q \in & \mathbb{R}^{n \times d},~W_2 \in \mathbb{R}^{e \times d} \newline
        V \in & \mathbb{R}^{n \times d},~W_3 \in \mathbb{R}^{e \times d}
    \end{aligned}    
\]

위와 같이 나타낼 수 있다. Attention value를 구하는 여러 방법들 중 transformer에서 사용된 scaled dot product 과정을 소개하면 다음과 같다.   
   

가장 먼저, 서로 다른 input embedding에 대한 score를 계산한다. $S = Q \cdot K^\top$. $Q$의 각 row vector는 각각의 token embedding에 대한 값이고, $K^\top$의 각 column vector는 각각의 token embedding에 대한 key가 된다. 이를 내적하게 되면 $Q,~K$의 row vectors $r^q_i,~r^k_i \in \mathbb{R}^d$ $(i = 1,~2,~\cdots,~n)$ 에 대해 다음과 같이 표현할 수 있다.

\[
    QK^\top = \begin{bmatrix}
        r^q_1 \newline
        r^q_2 \newline
        \vdots \newline
        r^q_n
    \end{bmatrix}
    \cdot \begin{bmatrix}
        {r^k_1}^\top & {r^k_2}^\top & \cdots & {r^k_n}^\top
    \end{bmatrix}
\]
이렇게 계산된 값은 각 row vector끼리의 내적으로 구성되며, 이는 row vector의 dimension $d$ 값이 커질수록 커지는 구조가 된다.

\[
    S = QK^\top = \begin{bmatrix}
        {r^k_1}^\top r^q_1 & {r^k_2}^\top r^q_1 & \cdots & {r^k_n}^\top r^q_1 \newline
        \vdots & \vdots & \ddots & \vdots \newline
        {r^k_1}^\top r^q_n & {r^k_2}^\top r^q_n & \cdots & {r^k_n}^\top r^q_n
    \end{bmatrix}  
\]

따라서 안정적인 학습을 위해(gradient를 맞춰주기 위해) 위의 score를 dimension의 square root value로 나눠준다.

\[
    S_n = \frac{QK^\top}{\sqrt{d}} = \begin{bmatrix}
        ({r^k_1}^\top r^q_1)/\sqrt{d} & ({r^k_2}^\top r^q_1)/\sqrt{d} & \cdots & ({r^k_n}^\top r^q_1)/\sqrt{d} \newline
        \vdots & \vdots & \ddots & \vdots \newline
        ({r^k_1}^\top r^q_n)/\sqrt{d} & ({r^k_2}^\top r^q_n)/\sqrt{d} & \cdots & ({r^k_n}^\top r^q_n)/\sqrt{d}
    \end{bmatrix}  
\]

구한 score를 확률값으로 바꿔주기 위해 softmax를 취한다. Softmax function은 exponential 함수를 통해 $0 \sim 1$ 사이의 값으로 normalization 해주고, 가장 중요한 점은 특정 dimension으로의 합이 $1$이 될 수 있도록 해준다.

\[
    \begin{aligned}
        \text{Let }z_{ij} =& ({r^k\_j}^\top r^q_i)/\sqrt{d}, \newline
        softmax(z_{ij}) =& \frac{e^{z_{ij}}}{\sum\_{i=1}^n e^{z_{ij}}} \newline
        P =& softmax(S_n)
    \end{aligned}    
\]

이제 여기에 마지막으로 key에 대응되는 value를 곱해주면, 우리가 얻고자 하는 weighted value matrix를 얻을 수 있다.

\[
    Z = V\cdot softmax(\frac{Q \cdot K^\top}{\sqrt{d}})    
\]

<p align="center">
    <img src="transformer/008.png" width="200"/>
</p>

## Masked attention in decoder
Attention 연산은 모두 위에서 설명한 것과 거의 동일하다. 다만 decoder에서 encoder와의 attention을 할 때는 encoder에서의 결과를 토대로 key, value를 상정하고 decoder의 output을 query로 삼는다. 그러나 학습 시에 <U>RNN과는 다르게</U> input으로 tokenized sentence 전체를 넣어주다 보니 causality 문제가 생긴다. 이 문제는 다음과 같다.   
Decoder에서 input 문장에 대해 번역된 결과를 내보낼 때는 recurrent 구조를 가진다. 이는 **RNN based sequence to sequence model**과 **transformer** 모두 동일하다. 그렇기 때문에 학습 과정에서는 '번역 결과'를 알고 있어도 이를 사용하면 안된다.
<p align="center">
    <img src="transformer/009.png" width="800"/>
</p>
이 그림을 보도록 하자. 그림에서의 task는 'I want to buy a car'이라는 문장을 다른 언어로 번역하는 과정을 나타낸 것이다. Decoder에서 가장 먼저 <U>BOS</U>(Begin of Sequence를 의미하는 토큰)와 encoder에서의 output을 기반으로 첫 단어('<U>Ich</U>')를 예측한다. 그 다음 단어는 예측된 '<U>Ich</U>'와 encoder에서의 output을 기반으로 두번째 단어('<U>will</U>')을 예측한다. 그 다음 단어는 예측된 '<U>Ich, will</U>'과 encoder에서의 output을 기반으로 세번째 단어('<U>ein</U>')을 예측한다. 이런 식으로 진행된다.   
물론 위의 과정에서 현재 예측될 단어가 다음에 예측될 단어를 참고하지 못한다는 사실은 'inference'(테스트)에서는 항상 성립한다. 그러나 학습 시에는 이미 번역된 결과를 모두 알고 있고, 이를 supervision으로 삼아서 네트워크를 학습하기 때문에 <U>학습 과정에서의 decoder는 뒷 단어들을 참고할 수 없게</U> 해야한다.

<p align="center">
    <img src="transformer/010.svg" width="800"/>
</p>

연산 과정을 간단하게 표현하면 위와 같다. 맨 윗줄부터가 decoder의 input token에 대한 attention을 나타내고, mask에서 색칠된 부분이 value가 $1$, 그렇지 않은 부분이 value가 $0$으로 $n$번째의 embedding은 $n$보다 작거나 같은 attention weight만 참고할 수 있다.

## Multi-head attention
머신러닝 기법 중 '앙상블'이란 개념이나, 딥러닝 convolutional neural network에서 사용하는 kernel의 channel 수는 모두 비슷한 기능을 가진다. 그것은 바로 같은 feature map에 대해 <U>여러 가지 representation을 학습할 수 있다</U>는 것이다. CNN에서의 구조는 이미 이를 보장할 수 있는 network **width**라는 특성을 가지지만, transformer의 경우에는 attention layer가 여러 채널을 가질 수 없다는 문제가 생긴다. 이를 해결하기 위해 등장한 개념이 바로 <U>Multi-head attention</U>이다.

<p align="center">
    <img src="transformer/011.png" width="300"/>
</p>

개념은 상당히 간단하게도, attention을 할 수 있는 linear layer를 head의 개수 $h$만큼 늘려서 사용하겠다는 것이다. Attention weight를 연산할 수 있는 head를 여러 개 사용하여 계산한 뒤, 결과값을 head의 축으로 concatenate한 뒤 Linear 연산을 통해 원래의 dimension으로 맞춰준다. 예컨데 위에서 사용한 notation 그대로를 사용해보면, head index $h$에 대한 query, key, value linear operator $W_h^Q,~W_h^K,~W_h^V$와 multi-head linear operator $W_o \in \mathbb{R}^{hd \times d}$에 대해서,

\[
    \begin{aligned}
        Q_h =& W_h^QX,~K_h = W_h^KX,~V_h = W_h^VX, \newline
        Z_h =& V_h\cdot softmax(\frac{Q_h \cdot K_h^\top}{\sqrt{d}}) \in \mathbb{R}^{n \times d}, \newline
        Z_{concat} =& \text{Concat}(Z_1;~Z_2;~\cdots;~Z_h) \in \mathbb{R}^{n \times hd}, \newline
        Z_{output} =& W_o \cdot Z_{concat} \in \mathbb{R}^{n \times d}
    \end{aligned}
\]

다음과 같이 계산되는 구조다.

## Positional encoding
RNN 구조에서 없었던 내용 중에 causality는 여전히 존재한다. 그런데 그것보다 더 문제인 것은 각 임베딩 사이의 거리도 어느 정도 context를 판단할 때 고려가 되는 사항인데, 각 token을 sequential하게 연산하는 RNN은 이를 걱정하지 않아도 되지만 transformer의 경우에는 해당 constraint를 줄 수 있는 방법이 필요하다. 예를 들면 다음과 같은 문장이 있다고 생각해보자,

\[
    \text{I have a dog named Adam and my friend has a cat named Epoch}    
\]
한글로 번역하면, 나한테는 <U>'아담'이라는 이름을 가진 강아지</U>가 있으며, 내 친구는 <U>'에폭'이라는 이름을 가진 고양이</U>가 있다는 뜻이다. 여기서 중요한 점은 강아지를 가지고 있는 주체와 고양이를 가지고 있는 주체에 대한 문장에서의 거리다. 만약 특정 이름과 동물이 연관된다는 특성에 대해서 token의 위치와 무관하게 학습이 진행되다보면, 다음과 같은 참사가 발생한다.   
   
'I have **a dog** named Adam and my friend has a cat named **Epoch**'   
'I have a dog named **Adam** and my friend has **a cat** named Epoch'   
   
여기서 <U>굵게 표시된 부분끼리 만약 attention value가 높게 책정이 되면</U>, 내가 키우는 강아지와 친구가 키우는 고양이의 이름이 뒤바뀌는 일이 생기는 것이다. 혹은 내 친구가 강아지를 키우고 내가 고양이를 키우는 식으로 번역이 될 수도 있다. 따라서 CNN에서 object가 서로 비슷한 픽셀 위치에 위치한다거나, translation에 불변한다는 특징과 같이 학습하는 과정에서의 inductive bias를 어느 정도 주기 위해서 token의 위치에 따른 임베딩을 추가해주는 방법을 고안하였다.
<p align="center">
    <img src="transformer/013.png" width="400"/>
    <img src="transformer/014.png" width="400"/>
</p>
Positional encoding으로 사용된 함수는 흔히 알고 있는 sinusoidal 함수를 사용한다. 

\[  
    \begin{aligned}
        PE_{(pos,~2i)} =& \sin (pos/10000^{2i/d_{model}}) \newline
        PE_{(pos,~2i+1)} =& \cos (pos/10000^{2i/d_{model}})
    \end{aligned}
\]

여기서 $i$는 embedding dimension을 따라가는 축이고, $pos$가 각 token의 위치를 나타내는 position이다. 이를 실제로 시각화한 그림이 좌측에 있는 얼룩덜룩한 형태가 된다. 식을 보면 알 수 있듯이 $d_{model}$은 앞서 언급했던 embedding의 차원 수 $d$와 같으며, $i$가 <U>증가할수록</U> sinusoidal 함수의 <U>주파수가 점차 감소</U>하는 모양새가 된다. 주기함수를 보면 알 수 있듯이 y축을 따라가면서 함숫값을 보게 되면 **같은 함숫값을 가지는 index**가 reference 위치와는 관계없이 동일한 것을 알 수 있다. 즉 <U>짙은 파란색이랑 또다른 짙은 파란색 사이의 y축 거리</U>(pos 차이)나, <U>짙은 빨강이랑 또다른 짙은 빨강 사이의 y축 거리(pos 차이)</U>가 일정하게 나타난다는 것이다. 그리고 또한 위의 함수는 각 embedding position마다 서로 다른 encoding을 더해줄 수 있다. 함숫값은 주기 함수에 의해 반복이 되지만, 그 어떤 encoded embedding도(x축에 평행한 value들을 의미) 서로 <U>일치하지 않고 unique</U>하다. 즉, 문장이 얼마나 길어지든 상관없이 각 token에 유일한 값으로 매핑할 수 있기 때문에, <U>문장의 길이와 무관하게 모든 token을 구분할 수 있다</U>.
<p align="center">
    <img src="transformer/015.png" width="400"/>
</p>
이처럼 구해준 각 token 별 encoding을 dimension에 맞게 생성해서 모든 embedding에 더해주면 된다.

## Transformer 학습

학습은 간단하게 supervision이 있기 때문에 cross-entropy loss에 대해 최적화가 가능하다. Output으로 나오는 softmax prediction에 대해 가장 최댓값을 가지는 index의 단어를 매핑하고, 제대로 된 단어의 확률값이 1에 가까워질 수 있게 학습된다.

## 그래서 요즘은?
Transformer는 단연코 neural network 중 최근에 가장 활발히 연구된다고 해도 과언이 아닐 정도로 다양한 분야에 접목하기가 좋다. 2017년에 기계 번역 task에 대해서 처음으로 transformer가 제시된 이후, BERT, GPT-3 등등 NLP 관련 모델들이 많이 등장했으며, 특히나 vision 분야에서도 vision transformer가 등장하면서 <U>multimodal approach</U>가 가능해졌다. 모든 modality에 대해 일관적인 구조로 encoding이 가능하다면 이를 토대로 여러 modality를 함께 supervision으로 사용할 수 있다는 관점이다.
<p align="center">
    <img src="transformer/012.png" width="600"/>
</p>

---

# Vision transformer

앞서 기계 번역에서 <U>transformer</U>의 구조 및 학습 방법에 대해 알아보았다. 따로 여기서는 첨부하지는 않았지만 실제로 transformer를 사용했을 때의 성능은 굉장히 좋았고, 이후 다양한 tansformer based approach를 통해 많은 <U>NLP 관련 기술들</U>이 발전할 수 있었다. 이에 비전에서도 NLP에서 사용된 transformer 구조를 사용할 수 없을지에 대한 연구가 'Attention is all you need' 논문 이후로 활발하게 진행되기 시작했다.   
하지만 명확하게 <U>tokenize되기 쉬운</U> natural language나 audio(단어나 음절 단위로 끊는 경우)와는 다르게 image의 경우에는 보다 연속적인 신호로 구성되어 있어서 정확히 어떠한 기준으로 tokenize를 해야할 지 애매하고, 이를 임베딩하는 방법 또한 일반적인 word2vec과 같은 방법을 사용할 수 없다는 점이 문제가 되었다.
<p align="center">
    <img src="transformer/017.png" width="400"/>
</p>
이러한 과정에서 처음으로 vision transformer의 성능을 보여준 논문이 [ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)라는 논문이었고, 해당 논문을 간단하게 살펴보면 다음과 같다. 논문에서 접근한 방식은 상당히 간단하다.
<p align="center">
    <img src="transformer/vit.gif" width="600"/>
</p>
이미지를 총 $n$개의 patch로 나눈다. 논문 제목에서 확인할 수 있겠지만, 이미지 크기가 $256 \times 256$ 이라면 이를 균등하게 $16 \times 16$의 patch로 분리한다. 위의 그림에서는 이를 좀 더 간단하게 표현하기 위해 이미지의 $H,~W$를 각각 3등분하여 9개의 샘플을 만들어내는 것을 볼 수 있다. 분리한 patch를 각각 flatten해서 1차원의 텐서로 만든 뒤 linear projection을 통해 embedding space로 보내고, 이 patch sequence 앞쪽에 <U>cls token</U>(이미지의 클래스를 구분할 때, global inference를 하기 위해 기준점이 되는 부분이라고 기억해두자)을 추가해준다. 앞서 transformer에서 했던 것과 같이 마찬가지로 여기서도 positional embedding을 더해주는데, 이미지의 경우 natural language embedding에서 사용했던 것과 같은 sinusoidal embedding의 효과가 나타나지 않기 때문에 여기선 그대신 <U>learnable positional embedding</U>을 사용하게 된다. 모든 attention 연산이 끝난 뒤 encoder output에서 <U>cls token</U>을 받아오고, 여러 번의 attention 연산이 진행되면서 이 cls_token에는 16개의 patch embedding에 대한 <U>global reference</U>가 완료된 상황이다. 이 cls token은 $B \times D$ 크기를 가지며, 이를 바로 class 예측에 사용하기보다는 attention 연산이 진행되면서 '<U>다른 패치들의 정보를 요약</U>'한다는 느낌으로 접근한다. 따라서 class embedding을 추출한 다음에는 다음과 같은 연산이 진행된다.

<p align="center">
    <img src="transformer/018.png" width="600"/>
</p>

16개의 patch가 있었기 때문에 transformer model의 $Q,~K,~V$ dimension $d$에 대해서 output은 cls token과 함께 $(16+1) \times d$ 크기의 tensor로 추출된다. 여기에 MLP head로 class 개수만큼의 output을 내보내는 linear 연산($d \rightarrow C$ where $C$ is the number of classes)을 진행한 뒤, layer normalization 결과를 <U>class 예측에 대한 score map으로 사용</U>한다. 즉, 이를 logit으로 삼아 softmax 연산을 하게 되면 비로소 cls token의 output에 대한 class 예측이 진행되는 것이다.

<p align="center">
    <img src="transformer/019.png" width="800"/>
</p>

그러나 vision transformer의 경우 데이터셋이 충분하지 않으면 BiT(ResNet 기반 transformer)와 같은 CNN 기반 모델에 비해 성능을 기대하기 힘들었는데, 이 이유는 바로 앞서 설명했던 inductive bias의 부재와 관련된다. CNN based network는 적은 데이터셋으로도 기본적으로 가지고 있는 inductive bias를 통해 빠른 일반화 및 최적화가 가능하지만, 패치 단위로 나누어 각각에 대한 attention을 연산하는 구조인 <U>vision transformer는 이런 의존성이 전혀 없기 때문</U>이다.   
그렇기 때문에 가지고 있는 단점이자 장점은, CNN의 경우 inductive bias 때문에 정해진 연산을 통해 추출할 수 있는 representation map 혹은 feature map이 어느 정도 학습이 진행되고 나면 수렴이 발생하지만, vision transformer는 데이터셋을 대량으로 학습시키면 학습시킬수록 attention 성능을 높일 수 있고, 이는 특정 구조에 따라 수렴하는 형태가 아니라 끊임없이 발전할 수 있다고 해석하였다.   
그렇기 때문에 Vision transformer를 <U>대량의 데이터셋</U> JFT dataset($3 \times 10^8$ samples)로 사전 학습한 뒤, 원하는 **downstream task**에 맞게 fine-tuning을 하면 더 좋은 성능을 보일 수 있다고 한다.

<p align="center">
    <img src="transformer/020.png" width="800"/>
</p>

그리고 <U>convolution</U> 연산에 비해 <U>attention</U> 연산이 가지는 parameter의 개수나 네트워크의 전반적인 규모가 CNN보다 <U>훨씬 크다</U>는 점이 문제가 될 수 있다. VGG-16의 경우 경량화가 많이 부족한 모델임에도 불구하고 175.12M의 parameter 개수를 가지기 때문에 ViT-Large 혹은 ViT-Huge에 비해 훨씬 가벼운 것을 확인할 수 있다. 심지어 ResNet의 경우에 가장 레이어가 깊은 네트워크 중 하나인 <U>ResNet-152</U>의 경우에도 paramter 개수가 <U>60.34M</U>에 그치는 것을 보면, 확실히 transformer 네트워크의 규모가 지나치게 크게 구성된 것을 볼 수 있다. 

---

# 왜 Vision transformer가 CNN을 이길 수 있었을까?
<p align="center">
    <img src="transformer/021.png" width="800"/>
</p>
가장 간단하게 풀자면 단순히 self-attention 이라는 연산의 mechanism이, 초기의 layer부터 입력된 input 전체에 대한 inference가 가능하기 때문이다. 예를 들어 CNN의 경우에는, $3 \times 3$의 커널 크기를 통해 $32 \times 32$ 크기의 receptive field를 가지기 위해서는

\[
    3+2 \times (n-1) \ge 32,~n > 14    
\]

14보다 많은 개수의 layer가 필요하다. 이 과정에서의 연산량을 줄이기 위해 추가적으로 max-pooling과 같은 filtering이 들어가면, 이미지의 특정 부분에서는 <U>global한 fine detail을 알아채기도 전</U>에 **image classification**을 해야 하는 문제가 발생하는 것이다. 이를 다르게 말하면 ViT는 모든 레이어에서 균일한 형태의 representation map을 획득할 수 있고, self-attention 연산을 통해 보다 빠르게 global information을 축약해낼 수 있다. 이렇게 빠르게 요약한 global 정보를 여러 레이어를 통해 각 패치마다 공유하면서, 잘 학습된 representation을 전달할 수 있게 되는 것이다. 그러나 어김없이 <U>dataset의 개수가 대량으로 필요하다는 조건</U>(약 10억 개의 샘플)과 그만큼 <U>parameter 수도 많이 필요</U>하다는 것이 제약 조건이 될 수 있다. 추가적으로는 high-resolution image의 경우 attention 연산이 $Q,~K,~V$에 대한 attention score를 연산하는 과정이 되기 때문에 multi-head self attention의 computation이 $(H \times W)^2$에 비례해서 증가하는 가장 큰 문제가 발생한다. 이제부터 그 각각의 문제점을 해결하기 위한 연구들을 소개해보겠다.

---

# Dataset을 효율적으로 사용할 수 있는 방법

---

# High resolution image에 대한 효율적인 연산

---

# CNN과 ViT을 어떻게 하면 잘 조합할 수 있을까?

---

...작성중