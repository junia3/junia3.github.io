---
title: InfoBatch - Lossless Training Speech-Up By Unbiased Dynamic Data Pruning 논문 리뷰
layout: post
description: Data pruning, Efficient Training
use_math: true
post-image: https://github.com/junia3/junia3/assets/79881119/b1cc9863-c72b-47b9-a324-0baac95d9360
category: paper review
tags:
- Data pruning
- Training time
- Efficiency
---

# 개요 및 문제

**Large-scale dataset(대용량의 데이터셋)**을 활용하는 **딥러닝 알고리즘**은 상당한 발전을 이루고 있지만,

**(1) 학습 시간이 오래 걸린다는 점**, **(2)** 학습 시간을 줄이기 위해서는 **Resource(리소스)가 많이 든다는 점**에서 활용하기 어렵다. 따라서 이를 해결하기 위한 여러 방법들이 소개되었다.

### **절대적인 샘플 수 줄이기**

가장 대표적으로 생각해볼 수 있는 방법은 원래의 데이터셋을 **그대로 활용하는 것과 엇비슷한 성능을 낼 수 있는** 상대적으로 작은 용량의 데이터셋을 구하거나 합성해내는 것이다. 소규모 데이터셋 집합을 **Synthesize(합성)**하는 **Dataset distillation**과 **Choose(선택)**하는 **Coreset selection**이 대표적인 방법에 속한다. 그러나 이 방법들은 소규모 데이터셋을 구성하는 과정에서 많은 리소스와 시간이 소모되며, 성능 하락을 피할 수 없다는 문제가 있다.

### **모델 수렴에 도움이 되는 샘플 위주로 샘플링**

데이터셋 중 상대적으로 모델 학습에 **도움이 되는 샘플**과 **도움이 되지 않는 샘플**로 구분이 가능하다고 가정하면(각각 수렴 속도가 빠르고 느리다는 것과 대응된다), 이들 모두를 동등한 확률로 샘플링하여 독립 항등 분포(*i.i.d.*) 배치 단위로 학습에 활용되는 것보다 빠른 수렴 속도에 기여할 수 있는 샘플을 더 높은 확률로 뽑는 방법을 생각해볼 수 있다. 이를 **Weighted sampling**이라 부르지만, 데이터셋과 모델 구조의 변화에 따라 성능 차이가 심한 문제가 있다.

### **큰 배치 사이즈로 학습**

리소스가 한정되어있지 않다고 해서 배치 사이즈를 **무작정 키우는 것**이 모델 수렴에 도움이 되지 않는 경우도 있다. 이러한 문제를 해결하기 위해 배치 크기가 큰 세팅에서의 학습을 위한 **LARS, LAMB**와 같은 **Optimizer(최적화 알고리즘)**이 제안되었다. 하지만 앞서 말했던 바와 같이 **한정된 리소스 때문에** 무작정 배치 사이즈를 키울 수는 없다.

### **Iteration 수 줄이기**

**Dataset pruning**은 학습 시 **최적화에 사용되는 샘플 수를 줄인다는 점**에서 Coreset selection과 크게 보면 비슷하지만, Coreset selection의 경우 전체 데이터셋을 대표하는 소규모의 데이터셋을 구성하는 것이 목적이라면, 이 방법은 **정보량이 높은 샘플들을 학습에 활용한다는 점**에서 차이가 있다.
정보량에 대한 필터링은 각 샘플의 score를 예측하는 방법을 사용하게 되는데, 정보량이 적은 샘플을 학습 과정에서 아예 제외하는 방식**(static pruning)** 방법은 제대로된 score 예측을 위해 데이터셋 전체에 대해 사전 작업 과정인 trial의 연산량이 높다는 단점이 있다. 이는 원래의 데이터셋의 크기가 커질 수록 더 심각한 문제가 된다. 따라서 정보량이 적은 샘플을 학습 과정에서 유동적으로 제외하는 방식**(dynamic pruning)** 방법이 제안되었고, 연산량을 줄이기 위해 **얻기 쉬운 logit, loss 혹은 gradient를 활용**하여 사전 작업 과정인 trial을 없애고 학습 과정에 pruning을 통합시킬 수 있었다.

앞서 소개한 여러 방법 중 **InfoBatch**는 4번 approach 중 **dynamic pruning**을 다루게 된다. 하지만 단순한(naive approach) dynamic pruning은 **학습이 치우치는(gradient biasing) 문제**가 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/b6021859-0b9c-49e0-b368-71a6b5d4c6dd" width="">
</p>

그림에서 보여지는 **EL2N Score**는 샘플 별로 예측된 확률과 실제 확률(one-hot vector)간의 차이에 대한 L2-Norm을 수치화한 값이다. 상대적으로 실제 확률과 차이가 큰 샘플들이 큰 loss value를 가질 것이고, 학습에 활용되었을때 모델의 최적화와 빠른 수렴에 도움이 될 것은 어느 정도 직관적인 결과로 예측된다. 하지만 해당 score를 기준으로 모든 샘플을 pruning하는(EL2N score 임계치보다 작은 샘플을 모두 학습에서 제외) 방법은 gradient biasing 문제를 가져오며, gradient biasing이란 원래의 데이터셋을 사용했을 때 모델이 최적화되는 방향과 다른 방향으로 모델이 학습되는 것을 의미한다. 즉, 단순한 data pruning로는 loss의 크기에 대해서는 고려해줄 수 있지만 **gradient의 방향을 제대로 고려해줄 수 없다는 것**이다.

---

# Contribution

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/2dbd05fb-0168-43f8-a56a-4c2dcee07330" width="">
</p>


따라서 논문에서 제안한 방법은 위와 같다. 각 샘플별로 구한 score를 기준으로 임계치보다 작은 샘플들을 전부 pruning하는 것이 아닌, 이들 중 **랜덤 샘플링을 통해 적은 갯수의 샘플만 학습에 활용**하게 된다. 또한 EL2N Score를 사용하지 않고 forward propagation을 통해 직접 구해지는 **loss value를 score로 활용**한다. 랜덤 샘플링을 통해 score가 임계치보다 작은 샘플들의 gradient를 보존할 수 있고, 샘플 수에 의한 gradient bias는 **보존된 gradient를 rescaling함으로써 조정**할 수 있다. 논문의 Contribution을 다음과 같이 정리해볼 수 있다.

- 기존 dynamic batch 방법이 가지고 있던 gradient bias 문제를 해결하였다.
- EL2N은 pruning cycle마다 소팅에 $O(\log N)$ 만큼의 연산(시간)량이 요구되었으나, 이 방법의 경우 소팅 과정이 불필요하므로 $O(1)$의 연산(시간)량으로 pruning이 가능하다.
- 데이터셋을 전부 사용하지 않고도 성능을 유지할 수 있었으며, classification, segmentation, SSL, LLM 등 다양한 task에 적용 (plug-and play)이 가능하다.

---

# Method

### Previous approach

이전 static/dynamic pruning 접근법의 문제점과 InfoBatch 방법을 보다 디테일하게 보면 다음과 같다.

원래의 데이터셋인 $\mathcal{D} = \{z_i = (x_i, y_i)\}_{i=1}^{\vert \mathcal{D} \vert}$와 각 샘플마다 정의되는 score $\mathcal{H}(z)$가 있을때, static pruning은 다음 조건과 같이 특정 score 임계치 $\bar{\mathcal{H}}$보다 작은 모든 샘플을 학습에서 제외시킨다.

\[
\mathcal{S} = \{z \vert \mathcal{H}(z) \ge \bar{\mathcal{H}}\}
\]

이를 training step 단위로 진행하는 방식이 dynamic pruning이다. Score $\mathcal{H}_t(z)$ 가 시간에 따라 변할 수 있다.

\[
\mathcal{S}_t = \{z \vert \mathcal{H}_t(z) \ge \bar{\mathcal{H}}_t\}
\]

두 방식 모두 학습 시 $\mathcal{S}$ 혹은 $\mathcal{S}_t$에 **포함되지 않은 샘플들을 제외시킨다**는 공통점이 있지만, dynamic pruning이 학습 과정에서 전체 데이터셋에 참조가 가능하기 때문에 gradient bias 문제는 상대적으로 적을 수 있다. 하지만 pruning cycle마다 소팅하여 샘플링한다고 하더라도 **low-score sample(학습에서 제외되는 샘플)이 지속적으로 겹칠 수 있기 때문**에 bias되는 문제는 근본적으로 해결할 수 없다.

Biasing 문제와 더불어 **데이터셋  전체를 참조할 수 없게될 확률이 높아진다는 것**은 전체 데이터셋을 학습에 활용할 때보다 성능 하락 문제를 가져오며, 매번 pruning 과정에서 scoring / pruning을  진행하기 때문에 데이터셋 크기가 커질수록 학습 시간에 영향을 미치는 것을 확인할 수 있다.

### InfoBatch

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/d739728f-c050-45e5-b64e-fd8b3344bcf9" width="">
</p>

일반적인 dynamic pruning과는 다르게, 임계치보다 작은 score를 가지는 샘플들 중에서 **랜덤하게 추출된 샘플만 pruning**하게 된다. InfoBatch에서는 하이퍼파라미터인 pruning probability $r$에 대해 다음과 같은 pruning policy를 적용한다.

\[
\mathcal{P}_t(z) = \begin{cases}
r,&\mathcal{H}_t(z) < \bar{\mathcal{H}}_t \newline
0,&\mathcal{H}_t(z) \ge \bar{\mathcal{H}}_t
\end{cases}.
\]

**임계치보다 score가 큰 샘플**은 $100\%$ 모두 사용하고, **임계치보다 score가 작은 샘플**은 $100(1-r)\%$ 만 사용한다. 또한 Forward propagation에서 계산된 loss 값의 평균을 임계치로 사용하게 되어 추가 연산 및 소팅하는 과정없이 pruning할 수 있다. Loss 값을 기준으로 삼은 이유를 저자는 두 가지로 밝혔다.

- **Extra-cost (추가 연산)없이** 바로 구할 수 있기 때문이다.
- Loss 값이 **각 샘플의 learning status(학습 정도)를 대표**할 수 있기 때문이다.

샘플의 score에 해당되는 $\mathcal{H}_t$는 매 epoch마다 업데이트되는데, 이때 $t$번째 epoch에서 학습에 관여되지 못한 샘플들(pruning된 샘플)은 optimization에 사용되지 않았기 때문에 score를 업데이트하지 않고, 학습에 관여한 샘플들($\mathcal{S}_t$에 속한 샘플)은 $t$번째 계산된 loss로 score를 업데이트하게 된다.

\[
\mathcal{H}_{t+1}(z) = \begin{cases}
\mathcal{H}_t(z),&z \in \mathcal{D} \backslash\mathcal{S}_t \newline
\mathcal{L}(z),&z \in \mathcal{S}_t
\end{cases}
\]

첫번째 epoch 학습 시에는 이전 학습 결과가 없기 때문에 score를 $1$로 초기화한 상태로 시작한다.

### 이론적 배경과 rescaling 방법

앞서 설명한 내용은 InfoBatch에서 기본적으로 사용한 dynamic pruning 방법에 대한 리뷰였고, 실제로 **랜덤하게 pruning된 dataset을 제대로 활용**하기 위한 이론적 내용을 정리하면 다음과 같다.

모든 딥러닝 모델은 학습하고자 하는 loss function $\mathcal{L}$을 가지고 있으며, 만약 연속 확률 밀도 분포 $\rho(z)$에서 추출되는 모든 데이터셋 $\mathcal{D}$를 사용하여 모델을 학습할 경우, 다음과 같이 **loss function의 전체 평균을 최소화**하는 모델 파라미터 $\theta \in \Theta$를 찾고자 한다.

\[
    \underset{\theta \in \Theta}{\arg \min} \underset{z\in\mathcal{D}}{\mathbb{E}} (\mathcal{L}(z, \theta)) = \int\_z \mathcal{L}(z,\theta)\rho(z)dz.
\]

앞서 설명한 pruning을 적용하면 score가 임계치보다 낮은 샘플에 대해서는 $1-r$ 만큼 normalized(혹은 scaling)된 확률 밀도 분포인 $(1-r)\rho(z)$, score가 임계치보다 높은 샘플에 대해서는 원래 확률 밀도 분포인 $\rho(z)$를 따르는 데이터셋을 사용하게 된다. 각각의 케이스에 대한 확률 밀도 분포를 통합하여 $(1-\mathcal{P}_t(z))\rho(z)$ 로 나타낼 수 있다.

Empirical한 관점에서 **pruning이 적용된 확률 밀도 분포에 대한 loss는 그만큼 손해를 봤기 때문**에 이를 **renormalize(혹은 rescaling)**을 해주기 위해 각 샘플에 대한 loss에 확률 밀도 분포에 곱한 factor인 $(1-\mathcal{P}_t(z))$의 역수를 곱하게 된다. 이렇게 곱해지는 역수 term을 $\gamma_t(z)$라고 하면, pruning 이후 남은 데이터셋인 $\mathcal{S}_t$에 대해서는 다음과 같은 **weighted loss function의 전체 평균을 최소화**하는 모델 파라미터 $\theta \in \Theta$를 찾고자 한다.

\[
    \begin{aligned}
    \underset{\theta \in \Theta}{\arg \min} &\underset{z\in\mathcal{S}\_t}{\mathbb{E}} (\gamma\_t\mathcal{L}(z, \theta)) = \frac{1}{c\_t}\int\_z \mathcal{L}(z, \theta)\rho(z)dz \newline
    c_t =& \mathbb{E}_{z \sim \rho}(1-\mathcal{P}\_t(z)) = \int_z \rho(z)(1-\mathcal{P}\_t(z))dz,~c\_t \in (0, 1)
    \end{aligned}
\]

또한 scale factor $\frac{1}{c_t}$는 데이터셋 샘플 갯수(혹은 iteration의 비율)인 $\frac{\vert\mathcal{D}\vert}{\vert\mathcal{S}_t\vert}$에 근사하게 되고, 따라서 **gradient가 샘플 pruning에 의해 감소된 만큼 보상받을 수 있다**고 가정할 수 있다.

\[
\mathbb{E}(\nabla\_\theta \mathcal{L}(\mathcal{S}_t)) \simeq \frac{\vert \mathcal{D}\vert}{\vert \mathcal{S}\_t \vert} \mathbb{E}(\nabla\_\theta\mathcal{L}(\mathcal{D}))
\]

### Gradient bias from pruning(Annealing)

만약 **상대적으로 초반에 pruning된 샘플**의 경우, 이후 학습 과정에서 랜덤하게 추출될 확률이 있으나, **후반에 pruning된 샘플**의 경우 이후 학습 과정에서 다시 추출될 확률이 줄어든다.

Score가 낮은 샘플 중 초/중반에 pruning된 샘플의 경우 운이 좋지 않아 샘플링될 확률에 들지 못했음에도 불구하고, 학습 후반이 되면서 점점 pruning될 확률이 높아지게 된다. 이는 학습 후반에 가까워지면서(revisiting 확률이 줄어들면서) 초반에 샘플링된 gradient에 bias될 확률이 높아지는 결과를 가져오게 된다. 따라서 저자는 **일정 epoch까지는 pruning을 통한 학습**을 진행하다가, **학습 후반에는 전체 데이터로 학습하는 방법**을 고안하였다. 전체 epoch을 $C$라고 했을 때, $1$에 가까운 하이퍼파라미터인 $\delta$를 조건에 추가해주게 된다.

\[
\mathcal{P}_t(z) = \begin{cases}
r,&\mathcal{H}_t(z) < \bar{\mathcal{H}}_t \wedge t < \delta \cdot C\newline
0,&\mathcal{H}_t(z) \ge \bar{\mathcal{H}}_t \lor t \ge \delta \cdot C
\end{cases}.
\]

이를 마지막으로 해석하면 다음과 같다.

1. 학습 epoch이 $\delta \cdot C$보다 작고(and) score가 임계치보다 작을 때 $r$의 pruning 확률을 적용하여 샘플링.
2. 학습 epoch이 $\delta \cdot C$보다 크거나 (or) score가 임계치보다 크거나 같을 때 전체를 샘플링.

---

# 주요 실험 결과

논문에서는 InfoBatch 방법이 다양한 방법에 적용 (plug-and play) 가능하다고 밝혔으며, 이를 실제로 다양한 벤치마크에 대한 실험 결과로 보여준다. 현재 **[아카이브에 있는 가장 최근 버전](https://arxiv.org/pdf/2303.04947.pdf)**을 기준으로 다음 데이터셋에 대한 Quantitative / Qualitative 결과가 제공된다. 데이터셋과 각각을 활용한 task는 다음과 같다.

- **CIFAR10/100, ImageNet1K** **:** Classification (이미지 분류)
- **ADE20K :** Semantic segmentation (이미지 세그멘테이션)
- **FFHQ :** Image generation (이미지 생성)
- **Instruction Dataset** : Instruction fine-tuning (LLM 미세 조정)

### Classification Benchmarks (CIFAR, ImageNet)

좌측 테이블은 **CIFAR 10/100**에 대한 정량적 평가, 우측 테이블은 **ImageNet1K**에 대한 정량적 평가에 해당된다.
<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/18a8e1ae-aa20-4929-8956-e1c368fe9dd2" width="700">
    <img src="https://github.com/junia3/junia3/assets/79881119/84937572-8ce8-4162-b07d-8d4f0c5a0665" width="500">
</p>


InfoBatch를 포함하여 기존 pruning 방식을 적용한 다른 approach에 대해 모두 비교한 모습이다. **CIFAR 벤치마크**의 경우 Fair comparison을 위해 동일한 pruning ratio에 대해 실험을 진행하였고, 30%의 pruning rate를 적용했을때 전체 데이터셋을 사용하는 것과 성능차이가 없는 것을 확인할 수 있다. **CIFAR 벤치마크**에서 InfoBatch 방법은 pruning ratio가 달라질 때마다 forward propagation number (학습 iteration) 수가 달라지기 때문에 이를 epoch number를 조정하여 같도록 맞추어 비교하였다고 밝혔다. **ImageNet 벤치마크**에서도 마찬가지로 다양한 모델 구조에 대해 실험을 진행하였고, 조정된 prune ratio에 대해 성능이 떨어지지 않고 유지되는 모습을 확인할 수 있다. **ImageNet 벤치마크**에 대한 학습 시간에 대한 효율성은 아래 그래프 및 표에서 확인해볼 수 있다. 학습 후반으로 갈수록 전체 데이터셋을 모두 활용하는 방식(baseline)과 비교하여 학습 시간이 훨씬 줄어드는 것을 확인할 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/57357d4f-1533-4cef-b63b-34c4055343ec" width="300">
    <img src="https://github.com/junia3/junia3/assets/79881119/9dce7285-459d-4e11-96dc-f118f3a3b298" width="800">
</p>


### Semantic segmentation (ADE 20k)

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/544c839d-de0f-4f2d-9bd1-54cfa13c883a" width="400">
</p>


**ADE20K 벤치마크**에 대한 결과는 위와 같이 나타났으며, 세그멘테이션 task의 경우에는 **mmseg 모듈을 사용**하게 되는데, 일반적인 task와는 다르게 **iteration을 기준으로 학습**을 하므로(epoch 단위가 아니라 데이터로더가 무한정 랜덤한 *i.i.d.* 샘플링이 가능하다고 가정하고, 최소 $40K$ 부터 최대 $160K$까지 학습한다), 연산 시간으로 비교하지 않고 목표 성능에 다다를 때까지의 iteration(index)로 비교하였다. 대략 $40\%$의 iteration 절감 효과가 있는 것으로 드러났다.

### Image generation (FFHQ)

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/c9139b76-42ec-4d85-b33d-2b36dc5760ba" width="500">
</p>


**FFHQ 벤치마크**를 사용하여 **LDM(Latent diffusion)** 모델을 학습했을 때 생성한 임의의 안면 이미지는 좌측과 같이 나타났고, 전체 데이터셋으로 학습된 LDM과 pruning을 통해 대략 $\sim27\%$의 학습량을 줄였음에도 이미지 퀄리티가 떨어지지 않았으며, 이미지 생성 task에서 활용하는 주요 평가 지표인 FID score 또한 좋은 모습을 보여준다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/a459ec50-9bbc-4e64-89ba-f2aee1776d8b" width="200">
</p>


### LLAMA Instruction fine-tuning (Instruction dataset)

LLM의 zero-shot 성능을 높이는 방법으로 **instruction dataset을 활용**하여 fine-tuning을 진행하는 연구가 진행되었고, 이를 **Instruction fine-tuning**이라 부른다. 저자는 앞서 제시한 computer vision task 이외에도 Instruction dataset을 활용한 **LLM task에도 InfoBatch가 활용될 수 있음을 보이는 실험**을 진행하였다. 표에 나타난 **DQ(Data Quantization)**를 적용하여 instruction dataset size를 1차적으로 줄이고, 이에 추가로 InfoBatch를 적용하여 학습하게 되면 학습 시간을 더 줄이고도 **평균 성능이 유지되는 것**을 확인할 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/94d6171f-ed9d-4eec-a6d4-461f529619e3" width="500">
</p>

---

# Ablation 실험

논문에서는 **실제로 적용한 각 방법**들이 효과적인지 확인하기 위해 ablation을 진행하였다. Ablation은 특별한 상황이 아니면 모두 **CIFAR100(Image classification) 벤치마크**를 사용하였다.

Random pruning(일반적인 dynamic pruning, hard pruning이 적용됨)을 적용했을때, 앞서 언급했던 biased gradient 문제 때문에 sub-optimal solution(저하된 성능)이 나타났으며, soft sampling을 적용하더라도 rescale(score가 낮은 샘플에 대한 gradient 크기 조정) 없이는 여전히 성능이 낮은 것을 확인할 수 있다. Rescaling만 적용하더라도 충분한 성능이 나오지만($78.1\%$), 후반 학습에 전체 데이터셋을 활용하는 annealing을 적용했을때 **전체 데이터셋을 모두 사용하여 학습했을때와 비교하여 같은 성능을 확보**($78.2\%$)할 수 있는 것을 확인할 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/460c9958-1e39-4d59-8fa4-8ce6debeab2f" width="500">
</p>


추가로 pruning condition에 대한 실험도 진행하였다. 원래대로 score가 낮은 샘플을 pruning하는 방식 이외에 score가 높은 샘플($\mathcal{H}_t(z) > \bar{\mathcal{H}}_t$)을 pruning하거나 랜덤하게 pruning하는 방법을 생각해볼 수 있다. 의외로 전반적으로 성능은 default setting ($\mathcal{H}_t(z) < \bar{\mathcal{H}}_t$)과 비교해서 큰 차이는 없는 것으로 나타났다. 그러나 prune condition을 다르게 설정하게 될 경우에 prune ratio를 줄이는 것이($33\% \rightarrow 16\%$) 상대적으로 original loss distribution을 크게 왜곡시키지 않을 수 있고(좌측 figure), 이는 실제로 **loss value 평균을 기준**으로 loss값이 **작은 샘플들이 차지하는 비율**(density)이 높기 때문에 pruning을 많이 하더라도 remained sample로 표현되는 **probability distribution이 원래 분포와 비슷할 것**이기 때문이다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/7d6f5e6f-4c28-44ea-8006-f3dc15cfadf3" width="350">
    <img src="https://github.com/junia3/junia3/assets/79881119/38f970fa-03c6-4193-8160-f16c00cbe3db" width="350">
</p>


추가로, loss 값이 작은 샘플들은 작은 gradient 값을 가지게 되고, 이를 사용하여 실제로 rescaling을 통해 **복구된 gradient의 분산은 원래 gradient의 분산을 upper-bound로 가진다는 사실**이 증명된다. Expectation과 비율을 생각하면, **더 안정적으로 원래 분포를 예측할 수 있게 된다**는 의미로 작용한다. 

\[
Var(G\_{\mathcal{S}\_t}) \le \frac{\vert \mathcal{D} \vert^2}{\vert \mathcal{S}\_t\vert^2}\mathbb{E}\_{z\sim D}(G^2\_z) -\frac{\vert \mathcal{D} \vert^2}{\vert \mathcal{S}\_t\vert^2}G^2 = \frac{\vert \mathcal{D} \vert^2}{\vert \mathcal{S}\_t\vert^2}Var(G\_D)
\]

따라서 실험 결과로는 비슷한 성능을 보였음에도, 저자가 주장하듯 **low-score sample에 대한 pruning이 보다 효과적**임을 알 수 있다. 이러한 분석을 사용하게 되면 **학습 안정성**까지 이어지는데, Loss 분포를 2차 Taylor expansion로 전개한 후의 **Hessian**과 **SGD optimizer의 관계**를 통해 확인할 수 있듯이 **gradient를 rescale**하는 것이 **step size의 rescale**로 이어지는 것을 확인할 수 있다. 이는 학습 불안정으로 이어지지만, 앞서 확인했던 바와 같이 variance가 rescale되는 경우, **step size의 rescale 효과를 상쇄시킬 수 있기 때문**에 안정적인 학습이 가능하다.

<p align="center">
    <img src="https://github.com/junia3/junia3/assets/79881119/914d8fb0-5502-4751-b8d9-1b4dfbf5b65f" width="500">
    <img src="https://github.com/junia3/junia3/assets/79881119/6eaf3382-f0bd-4732-bed7-b6cf9b83b353" width="600">
</p>


좌측 그래프는 학습 시 사용된 **하이퍼파라미터**인 $r$(pruning 비율)와 $\delta$(pruning을 진행할 최대 에폭 비율)에 대한 **CIFAR100 벤치마크** 실험 결과에 해당된다. 우측 표는 **CIFAR10 벤치마크**에 여러 optimizer를 사용했을때의 결과를 보여주며, optimizer에 상관없이 **InfoBatch를 사용**했을때와 **전체 데이터셋(full dataset)을 사용**했을때와 성능 차이가 거의 없는 것을 볼 수 있다.

실제 오피셜 깃허브를 참고하여 실험을 reproduction 해보았는데, 결과는 [깃허브 링크](https://github.com/junia3/InfoBatch)에 업로드하였다.