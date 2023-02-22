---
title: 딥러닝의 체계를 바꾼다! The Forward-Forward Algorithm 논문 리뷰
layout: post
description: prompt learning
use_math: true
post-image: https://user-images.githubusercontent.com/79881119/220503923-6b3857d3-ebc4-46ed-b2fe-270444a158bc.gif
category: paper review
tags:
- Deep learning
- FF algorithm
- Methodology
---

# 들어가며...

제목이 너무 어그로성이 짙었는데, 논문에서는 backpropagation을 <U>완전히 대체하는 알고리즘을 소개한 것은 아니고</U> 새로운 연구 방향성을 잡아준 것과 같다.

이 논문에서는 neural network를 학습하는 <U>기존 방법들</U>로부터 벗어나 새로운 학습법을 소개한다. 새롭게 제시된 방법인 <U>FF(forward-forward)</U> 알고리즘은 뒤에서 보다 디테일하게 언급되겠지만 Supervised learning과 unsupervised learning의 몇몇 간단한 task에 잘 적용되는 것을 볼 수 있고, 저자는 이를 통해 FF 알고리즘이 기존의 foward/backward 알고리즘과 더불어 더 많은 연구가 진행될 수 있을 것이라고 전망한다. 아마 딥러닝을 하던 사람들은 가장 기초부터 배울 때 backpropagation이라는 개념을 필수로 배울 수 밖에 없으며, 본인이 블로그에 작성한 글 중 신경망 학습을 위해 제시된 backpropagation이라는 개념을 perceptron의 역사와 함께 소개하는 내용이 있었다([참고 링크](https://junia3.github.io/blog/cs231n04)).   
기존 backpropagation 방법은 forward pass를 통해 <U>오차를 계산한 뒤</U>(supervision이 있다고 가정하면) backward pass 시 chain rule을 통해 각 parameter를 learning rate에 따라 업데이트했다면, forward forwad algorithm(FF)은 한 번의 <U>positive pass(real data에 대한 stage)</U>와 한 번의 <U>negative pass(네트워크 자체에서 생성되는 data에 대한 stage)</U>로 구성된다.   

---

# 논문에서 제시한 backpropagation의 근본적인 문제점
사실상 딥러닝은 큰 갯수의 parameter를 가진 model을 stochastic gradient 방법을 통해 대량의 데이터셋에 fitting하는 과정이었다. 그리고 gradient는 <U>backpropagation</U>을 통해 연산하게 되었다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220493453-18b6ebe8-cc5e-4316-932b-3ae8006f52db.png" width="400">
</p>

인간의 뇌가 작동하는 구조를 모방한 것이 신경망이었다. 시냅스가 서로 연결되어있으며 이전 신호의 magnitude에 따라 activation이 일어나는 방식으로 구성한 <U>다층 신경망 구조</U>는 실생활의 이런 저런 문제들을 해결할 수 있는 힘이 있었다. 이러한 발전이 있기 위해서는(다층 신경망을 학습시키기 위해서는) <U>backpropagation 알고리즘</U>이 필수적이었으며, 여기서 발생하는 의문점은 다음과 같았다.

- 인간의 뇌가 실제로 학습할 때 backpropagation 방법을 사용하는가?
- 만약 인간의 뇌에서도 학습할 때 backpropagation과 같은 방법을 사용하지 않는다면, 시냅스 간의 연결 사이에 가중치를 조절하기 위한 메커니즘이 따로 존재하는가?

## 피아제의 인지발달이론

잠시 논문을 소개하기 전에 심리학 이론에 대한 설명을 하고 넘어가도록 하겠다. 직접적으로 이 논문과 관련이 있을지는 모르겠지만, 논문을 읽으며 근본적으로 backpropagation에 의문을 가진 과정 자체가 인간이 어떤 <U>정보를 학습하는 메커니즘과의 차이</U> 때문이라고 생각했다. 
실제로 우리 뇌의 cortex(피질)를 생각해보면 backpropagation은 뉴런으로 구현할 수 있음이 증명되었지만 우리가 실생활에서 학습하는 방식과는 차이가 있다. 예를 들어 <U>어떤 아이</U>가 난생 처음으로 강아지를 본다고 생각해보자([참고 링크](https://www.simplypsychology.org/what-is-accommodation-and-assimilation.html)). Piaget(피아제)의 <U>인지발달이론</U>을 인용하자면, 인간에게 있어서 accommodation과 assimilation 과정이 반복되면서 인지 적응이 진행된다고 설명한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220494883-88e9a0a7-8bbf-4baf-8784-cc07bac1da9f.png" width="400">
</p>

아이의 보호자는 아이에게 강아지의 생김새에 대한 묘사를 해주거나, 그림책을 기반으로 '강아지'라는 존재에 대한 특징을 입력받는다.
아이는 기존에 강아지에 대한 어떠한 지식도 없었기 때문에 '강아지'라는 존재는 <U>본인의 인식 체계 속</U>에서 dissimilar한 존재다(낯선 존재). 따라서 강아지에 대한 <U>정보가 주어진 순간</U>에는 해당 지식에 대한 <U>불안정한 체계</U>가 잡히게 되고, 아이는 강아지에 대한 새로운 특징이나 정보를 입력할 때마다 '강아지'에 대한 인식 체계를 확립하며 이를 안정화하는 단계에 이른다.
이런 상황에서 길을 걷다가 실제로 <U>강아지</U>를 만났을 경우를 생각해보자. 아이는 본인이 안정화시킨(확립한) 강아지에 대한 특징을 토대로 목격한 대상을 강아지라고 판단하게 된다. 그런데 갑자기 강아지가 <U>예상치 못한 행동</U>을 하는 경우를 생각해보자. 강아지가 <U>'짖고', '물고', '핥고'</U>하는 특징들은 아이가 기존에 경험해보지 못했기 때문에 본인이 확립한 '강아지'라는 <U>인식 체계에 disequillibrium을 주는 특징</U>들이다. 이러한 혼란스러운 상황에서 아이는 부모나 정보를 제공해줄 수 있는 사람을 통해 '강아지가 맞다'라는 확답을 듣게 되면, disequillibrium 상태에 있었던 <U>지식 체계가 강화</U>된다(reinforcement). 이를 동화(assimilatioon) 과정이라고 부른다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220498798-01bddac3-658f-49cc-99c0-453f3da78280.png" width="400">
</p>

Accomodation은 조금 다르게, 처음 보는 존재를 분류할 때 본인이 인식하고 있는 특징과 <U>다른 점들을 통해 새로운 정보 체계를 확립</U>하는 과정이다. 예를 들어 길을 가다가 고양이를 본 경우를 생각해보자. 고양이는 강아지와 다르게 '야옹'하는 소리를 내고, '나무에 올라가거나' 등등 여러 다른 특징들을 보여준다. 기존에 강아지가 본인이 알고 있는 특징들과 다른 모습들을 보여준 경우에도 아이는 같은 <U>disequillibrium 과정</U>을 거쳤고, 이런 상황에서 정답을 제공해줄 수 있는 사람을 통해 <U>정보 체계를 강화</U>했던 것과 비슷한 방식으로 아이는 정보를 제공해줄 수 있는 사람에게 강아지가 맞는지 묻게 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220499468-238695a2-b377-40e4-8998-5a2eb773d618.png" width="400">
    <img src="https://user-images.githubusercontent.com/79881119/220499618-eb68d8de-a747-4e63-ba1a-d8d83a788940.png" width="400">
</p>

그러나 이번엔 강아지가 아니라 본인이 기존에 알지 못했던 정보인 '고양이'라는 대답을 듣게 되고, 이를 통해 본인이 <U>기존에 알고 있는 강아지에 대한 특징</U>에 새로운 존재인 <U>고양이의 특징</U>을 접목시켜 새로운 정보에 대해 적응하는 과정을 겪는다. 이러한 과정을 <U>accomodation</U>(적응)이라고 부른다.

## 그래서 인간의 학습 과정은?
피아제의 인지 발달 이론에 대해 굳이 짚고 넘어 온 이유는 인간은 본인이 <U>알지 못했던 사실</U>이나 <U>새로운 사실</U>을 받아들이는 과정에서 본인이 가진 인식 체계(일종의 뉴런 weight)가 도출한 잘못된 결과에 대해 오차를 계산한 뒤 이를 <U>다시 적용시키는 과정</U>이 <U>explicit하게 존재하지 않는다</U>는 사실이다. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220500694-89a1e6d3-ad86-4420-9744-62ff8c24eac4.png" width="400">
</p>
시각 정보를 처리하는 visual cortex가 연결된 구조는 top-down 형태로, 시각 정보를 받아들이는 <U>가장 바깥쪽 부분</U>부터 차례로 정보를 처리하게끔 되어있다. 만약 backpropagation이 진행되는 구조는 이와는 반대로 <U>가장 안쪽 cortex부터 망막까지 이어지는 시신경 세포들까지</U>의 bottom-up mirror 구조를 가져야하는데, 실제로는 그렇게 되지 않는다는 것이다. 오히려 우리가 보는 시각 정보는 연속적인 프레임을 가진 일종의 동영상이며, <U>잘못된 판단에 대한 ground truth가 주어졌을 경우</U>(강아지라고 했는데 사실은 고양이였을 경우) 이전에 관찰한 시각 정보에 대한 <U>nerve signal</U> 오차를 계산해서 역방향으로 정보를 학습하는 것이 아니라, 우리가 <U>지금 보고 있는 이미지에 대해</U> 정보 체계를 수정하게 된다. 즉 backpropagation 구조라기 보다는 시각 정보를 통해 신경 activity가 발생하는 내부에서 <U>하나의 루프를 생성</U>하고, 이 과정으로 <U>정보 체계를 바꿔가는 것</U>으로 볼 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220513070-eb6e8bb0-f122-4fe9-9da6-8dac3584a4d9.png" width="800">
</p>

그리고 만약 <U>우리가 학습하는 정보에 대해서도</U> backpropagation이 진행된다면 형광등이 빠른 속도로 점멸하는 것처럼 우리의 <U>인식 체계에도 주기적인 time-out</U>이 필요하다. 딥러닝에서 하는 일종의 <U>online-learning</U>과 비슷한데, 우리가 일상생활을 유지하면서 그와 동시에 backpropagation이 가능하기 위해서는 뇌의 각 단계에서의 sensory processing 결과를 저장할 pipeline이 필요하고, 이를 오차에 맞춰 수정한 뒤 원래의 인식 체계에 적용할 수 있어야 한다. 하지만 <U>pipeline의 뒤쪽에 있는 정보</U>가 backpropagation을 통해 <U>earlier stage</U>(보다 input에 가까운 위치)에 영향을 끼치지 위해서는 <U>실시간으로 인식을 진행하는</U> 우리의 학습 과정과는 차이가 있어야 한다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220514216-e547f003-4f09-4cf9-867a-3857f6f95a41.png" width="800">
</p>

Backpropagation은 또한 forward pass 과정에서의 <U>모든 연산 결과</U>를 알아야한다는 것이다. Chain-rule에 의해 각 노드에서의 <U>local gradient를 계산</U>하기 위해서는 노드에서의 input을 알아야하고, 이는 곧 이전 노드들의 output을 모두 알아야 가능하기 때문이다. 그렇기에 <U>forward pass 과정이 black box</U>라 가정하면(어떤 연산이 진행되는지 전혀 모른다고 생각하면), 미분 가능한 모델이 확립된 상황이 아니라면 <U>backpropagation이 진행될 수 없는 것</U>을 알 수 있다. 이를 바꿔 설명하자면 만약 인간의 인식 체계가 backpropagation을 적용하기 위해서는 시신경을 포함하여 판단을 내리는 모든 구조에 대해 <U>differentiable closed form</U>으로 알고 있다는 전제가 필요하다. 이러한 문제들을 forward-forward algorithm에서는 고려할 필요가 없다.

또다른 방법으로는 강화학습을 생각해볼 수 있다. Forward process에 대한 정보가 부재할 경우에는 단순히 neural activity에 대한 weight의 일부에 random한 변화를 가해주고, 변화에 따라 바뀌는 <U>결과값에 대한 보상</U>을 해주면 된다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220514711-1707850b-19dc-45ac-b9dc-55275351196a.png" width="400">
</p>

하지만 특정 parameter의 변화가 <U>다른 parameter의 변화에 종속</U>할 수 있는 기존의 backpropagation과는 달리, 강화학습의 경우에 <U>variance(경우의 수)가 너무 크기 때문</U>에 각 parameter의 변화가 output에 미치는 영향을 제대로 확인할 수가 없다. 이러한 문제를 학습 과정에서 생기는 noise라 하는데, 이를 완화하기 위해서는 변화가 가해지는 parameter의 개수에 반비례하게 learning rate를 구성하는 방법이 있다. 결국 <U>parameter의 개수가 증가할수록</U> 학습 속도는 이에 반비례해서 <U>계속 감소</U>하게 되며, 대용량의 네트워크를 학습시킬 수 있는 backpropagation 알고리즘에 비해 <U>학습 속도 측면</U>에서 불리하게 작용한다.

이 논문에서는 <U>ReLU나 softmax</U>와 같이 closed form으로 구할 수 있는 non-linearity를 포함하지 않는 네트워크도 학습할 수 있는 forward-forward algorithm(FF)을 제안한다. FF의 가장 큰 장점은 backpropagation 방법에서는 forward pass에 대한 <U>레이어 연산이 불명확한 경우</U>에는 <U>학습이 불가능</U>하다는 점을 해결할 수 있다는 사실이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220517775-fe18eb62-c188-4ba5-99cb-6478931496e6.png" width="1000">
</p>

또한 <U>연속된 데이터</U>가 주어졌을 때 다음과 같이 neural network의 output에 대한 <U>error를 통해 parameter를 업데이트</U>하는 과정에서 pipelining을 멈출 필요가 없다는 점도 장점이 될 수 있다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/79881119/220519115-745feb79-b4f3-4631-82ac-fbcdc1d680a7.png" width="400">
</p>

하지만 논문에서 밝히는 것은 backpropagation보다는 forward-forward algorithm이 <U>속도가 더 느리고</U> 실험한 몇몇의 toy problem 이외에는 아직 일반화가 힘들다는 문제가 있기 때문에 FF 알고리즘이 backpropagation을 완전히 대체하기는 힘들다고 밝힌다. 그렇기 때문에 여전히 대용량의 데이터셋을 기반으로 하는 딥러닝은 <U>backpropagation을 계속 사용할 것</U>이라고 한다.

---

# Forward forward algorithm이란?

...작성중