---
title: Github 프로필 리드미 작성하는 법 및 꾸미기 (feat. github readme snake)
layout: post
description: Profile, Developer
use_math: true
post-image: https://github.com/junia3/junia3.github.io/assets/79881119/09e4be19-b1c9-4f1a-856b-ad8c0c1555c3
category: development
tags:
- Github
- Readme
- Profile
- Portfolio
---

# 깃허브를 꾸며야 하는 이유
깃허브를 꾸며야만 하는 이유에 대해 설명하기에 앞서, 조금은 개인적인 이야기를 덧붙이고자 한다. 엔지니어라면 가장 중요한 역량은 이론에 기반한 공학적 지식과 더불어 이를 실제 산업 환경에서 적용할 수 있는 능력 그리고 새로운 기술 전반에 대한 거부감이 없어야한다는 점 등이 있을 것이다. 본인도 이렇게 흔히 생각할 수 있는 현실적 역량들이 최우선이라 생각했고, 그렇기 때문에 스스로를 PR하는 방식에 있어서 **프레젠테이션**이나 **멋드러진 포트폴리오**는 없다고 생각해왔다.  
이러한 생각은 여러 발표 경험, 그리고 새로운 사람들을 만나면서 자극을 받으면서 달라졌다. 애써 좋게 보이려는 노력 하나가 작게는 내 주변 사람들의 시선부터 시작해서 크게는 내가 잘 모르는 사람들에게까지, 그리고 더 중요하게는 내 자신의 발전을 위해서도 중요하다는 사실이었다.  
포트폴리오는 <U>나를 증명하는 수단</U>이며 나를 모르는 사람들에게는 내 <U>얼굴로 비춰진다.</U> 예컨데 내가 팔자에도 없는 프론트 지식을 익혀가면서까지 이 블로그를 열심히 꾸민 이유도, 이렇게까지 글을 열심히 써내는 이유도 마찬가지다. 나는 남들과 같은 그런저런 비슷한 깃허브 블로그는 만들고 싶지 않았고, 내 포트폴리오를 날림식의 공부처럼 남겨놓고 싶지는 않았다. 개발자, 혹은 컴퓨터를 전공으로 하는 사람들에게는 깃허브는 일종의 <U>포트폴리오 및 블로그, Social Network Service</U>이며 commit 하나를 얼마나 신중하게 작성하는가에 따라 미래의 나 혹은 나를 제외한 누군가가 과거의 나에게 도움을 받거나 과거의 나에게 증명받는 과정이 명확해진다.  
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/354580ce-f735-4cf0-aaf9-ff8204f96afb" width="400">
</p>

<U>깃허브 프로</U>필은 여느 SNS 계정과 비슷하게 사진, 사용자 이름 그리고 홈페이지 등으로 구성되지만 보다 중요하게 여겨지는 요소는 실제로 그동안 얼만큼 여러 project에 contribution을 했었고, 이러한 contribution을 github platform 상에 잘 정리했는가이다. 하지만 솔직히 말하자면 잔디를 심는 과정은 그냥 적당한 레포지토리에 작은 코드라도 이것저것 매일같이 올리거나 엄청난 프로젝트를 하지 않더라도 할 수 있다. 즉 어느 정도 <U>'깃허브'</U>를 타인 평가의 지표로 잘 사용하던 사람은 단순히 깃허브라는 플랫폼에 코드만 많이 올린다고 장땡은 아니라는 사실을 이미 알고 있을 것이다. 개인적으로 깃허브를 여전히 제대로 활용하고 있지 못하다고 생각하고, 그렇기 때문에 깃허브를 꾸며야하는 이유에 대해 작성하면서도 인지부조화가 일어나지만 적어도 앞으로 점차 중요성을 인지하고 꾸준히 발전해나가고자 마음을 다잡기 위해 이 글을 작성해보려한다. 그리고 내가 깃허브 프로필을 꾸민 방식, 테마에 대해 간략하게 정리해보고자 한다.

---

# 프로필 만들기
프로필을 만드는 법은 간단하게 자신의 깃허브 ID와 동일한 이름의 레포지토리를 만들고, 여기에 ```readme``` 파일을 만들면 된다.
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/34668121-b9d3-471a-98c7-8a6bd89cdf3b" width="300">
</p>
자신의 ID를 모르는 경우는 거의 없겠지만 혹시라도 기억이 잘 안나면 본인 계정 프로필 들어간 다음에 상단의 'https://github.com/' 뒤에 붙어있는 걸로 확인하거나 직접 프로필 상의 ID를 확인하면 된다. 내 ID는 ['junia3'](https://github.com/junia3)이므로 앞으로 글을 작성할 때 이 ID를 기준으로 작성할 예정이다. 아무튼 내 github ID는 junia3이므로 프로필용 repository를 만들 때 이 이름을 그대로 사용하면 된다.
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/467f8db9-4d66-42ba-aa0f-9c150cb2eeb7" width="900">
</p>

본인은 이미 만들어져있는데, 만약 없다면 오른쪽 상단의 ```new```를 눌러서 새로 만들어주면 된다. 들어가서 repository 이름을 그렇게 지정하면

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/6650600a-0e76-452a-b348-35e31643b17f" width="700">
</p>

아직 만들기도 전인데 열심히 캐릭터가 설레발치면서 좋아라한다. 암튼 이렇게 나오면 제대로 입력했다는 뜻임. 당연히 리드미 파일을 포함할거니까 

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/2d4c601e-5b97-47b0-995e-7efa038ee71e" width="500">
</p>

대강 이렇게 해두고 <U>create repository</U>하면 된다.

---

# 프로필 꾸미기

프로필을 구성할 수 있는 요소는 정말 다양한데, 그냥 내가 적용한 요소들에 대해서 간단하게 소개하겠다. 현재 본인 깃허브 프로필은 다음과 같이 구성되어있다. 

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/8512b62e-a516-4694-8837-091306f33ed7" width="500">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/937d65b7-05f0-4e59-8ac4-010aef306368" width="500">
</p>

본인은 전체적인 톤앤매너를 맞춰주는 걸 선호하고 좋아하기 때문에 깃허브 다크모드를 기준으로 색상을 맞춰보았다. 각 요소들을 디자인함에 앞서 사용법과 관련하여 디자인 세부 프레임워크를 나열하면 다음과 같다.

### 프로필 상/하단에 header/footer 배너 추가하기
프로필을 보면 상단과 하단에 그라데이션으로 색상 톤을 맞춰준 배너가 존재한다. 이는 흔히 많이들 사용하는 [capsule-render](https://github.com/kyechan99/capsule-render)를 사용하였고, 들어가면 사용법에 대해서는 디테일하게 나와있다. 예컨데 내가 사용한 배너는 상단은 ```waving```, 하단은 ```rect```이다. 배너에 직접 텍스트를 넣을 수도 있다.  
본인이 사용한 상/하단 코드는 다음과 같다. 위에서부터 차례대로 상단과 하단 배너에 대한 코드로, 본인은 리드미 전체를 중앙정렬된 div 태그로 감싸주었고 배너는 해당 태그 내부의 양 끝단에 위치한다.
```html
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:E34C26,10:DA5B0B,30:C6538C,75:3572A5,100:A371F7&height=100&section=header&text=&fontSize=0" width="100%"/>
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:E34C26,10:DA5B0B,30:C6538C,75:3572A5,100:A371F7&height=40&section=footer&text=&fontSize=0" width="100%"/>
```

컬리의 경우 gradient를 지정해줄 수도, 자동으로 하거나 단색 지정 등 여러 옵션이 있는데 본인은 조금 특별하게 깃허브 페이지 스타일 자체가 가진 색상과 어우러지길 원했기 때문에 개발자 도구(f12)를 사용하여 프론트를 구성하는 색상을 알아낸 뒤 이를 사용했다. 그라데이션 위치는 $0 \sim 100$ 까지를 배너의 가장 좌측부터 우측이라고 생각하면 되고, footer는 그 반대로 생각하면 된다.  
만약 0, 25, 50, 75, 100에 직접 색상을 지정한 gradient를 구현하려면 위의 코드에서,

```html
?color=0:color1,25:color2,50:color3,75:color4,100:color5
```

이 부분을 바꿔주면 된다. color1 ~ color5는 직접 hexcode로 뽑아서([참고 링크](https://www.color-hex.com/)) 사용하는걸 추천한다.

### CV(Resume) 추가하기

이 부분은 진짜 별거없는게 그냥 markdown 문법을 써서 간단하게 언급할 내용만 작성해놓았다. 직접 코드를 확인하는게 편한데, ([참고링크](https://github.com/junia3/junia3)). Markdown을 다 작성한 다음에 숨기고자 하는 부분을 detail 코드로 감추면 된다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/9b3a1db2-2a45-4cf4-9526-c16fb97cbdfa" width="700">
</p>

```html
<details>
<summary>About Me (접혔을 때 화살표 옆에 뜨는 텍스트)</summary>
어쩌구저쩌구 (감추고자 하는 내용)
</summary>
```

### CV에 들어가는 instagram, gmail, github blog 등등 여러 배너 추가하는 방법
또한 본인의 프로필에서 CV를 펼쳐보면 연락 수단이나 additional skill란에 배너가 추가되어있는데, 이를 적용하는 법은 다음과 같다. 우선 [shields.io 홈페이지](https://shields.io/)에 들어간다.
```Get Started```를 누르면 다음과 같이 static badge를 커스텀해서 해당 뱃지의 ```html``` 임베딩 코드를 빼내는 사이트가 나온다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/aafc87c3-6915-4bad-8d0f-b8116ae8b300" width="1000">
</p>

또한 코드를 가져왔을때, 배너에 로고가 들어가게끔 해주고 싶다면(아래 그림과 같이 instagram, gmail 등등 뱃지에 맞는 로고) 우선 [심플 로고](https://simpleicons.org/) 사이트에 들어간 뒤에,

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/da36a212-ddeb-443b-88a1-b07ed6f586fc" width="400">
</p>

찾는 로고 이름을 검색어에서 찾으면 엥간한 애들은 다 나온다(프로그래밍 언어도 다 나옴)

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/248ecb72-9f57-45f2-98a2-c670079636c4" width="600">
</p>

여기서 로고에 나와있는 hexcode 색상(#E4405F)이랑 그 바로 위에 있는 로고 이름(Instagram)을 기억한 채로 다시 배너 코드로 돌아와서,

```html
<img src="https://img.shields.io/badge/Instagram-hexcode색상?style=plastic&logo=로고이름&logoColor=로고색상"/>
```

이런 식으로 작성하게되면 배너는 hexcode 색상, 로고이름에 해당되는 로고가 로고색상에 맞게 들어간다. 여기에 추가로 실제 인스타그램 링크와 연결한다던지, 그리고 여러 배너를 옆으로 붙이는 방식은 다음과 같이 ```span``` 태그와 ```a``` 태그를 함께 적절하게 활용해주면 된다. 실제로 본인이 인스타그램 배지에 사용한 코드는 다음과 같다.

```html
<span>
  <a href="https://www.instagram.com/6unoyunr/">
    <img src="https://img.shields.io/badge/Instagram-ff69b4?style=plastic&logo=Instagram&logoColor=white"/>
  </a>
</span>
```

이를 전반적으로 쭉 사용하면 다음과 같이 꾸밀 수 있다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/9119ed6c-f1ca-4d7c-90be-ba2ca9279378" width="600">
</p>

### 깃허브 status 추가하기
<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/869b0c81-5b9d-4c75-8586-cb30ba0ee5a3" width="1000">
</p>

깃허브 프로필에 내가 자주 사용하는 언어, 깃허브 <U>총 commit/star 개수</U>나 <U>날짜별 contribution graph</U>를 보여주는 곳이다. 해당 UI는 모두 [github-readme-stats](https://github.com/anuraghazra/github-readme-stats)에서 구할 수 있는데, 본인은 해당 UI들에 대한 코드 중 일부 스타일만 수정하여 그대로 사용하였다. 사실 이 부분은 그렇게 별다른 설명이 필요하지 않을 정도로 복붙이라서 코드만 올리면 다음과 같다. 내 코드에서 username이라고 되어있는 부분이나 bg_color/icon_color/title_color을 수정해주면 되고, 개인적으로 특정 레포에서 투머치로 사용된 ```C#``` 언어가 most used language로 뜨는게 싫어서 해당 repository를 exclude해주었다. 그리고 높이랑 너비 맞춰주는게 예뻐서 ```width``` 부분은 노가다 뛰면서 맞췄다.


```html
<a href="https://github.com/anuraghazra/github-readme-stats">
    <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=junia3&layout=donut&show_icons=true&theme=material-palenight&hide_border=true&bg_color=20232a&icon_color=58A6FF&text_color=fff&title_color=58A6FF&count_private=true&exclude_repo=Face-Transfer-Application" width=38% />
</a>    
<a href="https://github.com/anuraghazra/github-readme-stats">
  <img src="https://github-readme-stats.vercel.app/api?username=junia3&show_icons=true&theme=material-palenight&hide_border=true&bg_color=20232a&icon_color=58A6FF&text_color=fff&title_color=58A6FF&count_private=true" width=56% />
</a>
<a href="https://github.com/ashutosh00710/github-readme-activity-graph">
    <img src="https://github-readme-activity-graph.vercel.app/graph?username=junia3&theme=react-dark&bg_color=20232a&hide_border=true&line=58A6FF&color=58A6FF" width=94%/>
</a>
```

### Github 잔디 먹는 뱀 만들기

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/25faad36-1317-4d4e-89a0-bfdcf87cfae3" width="1000">
</p>

사실 이 부분이 생각보다(?) 제일 애를 먹었는데, 애초에 생긴것부터 리드미에 이게 왜있지 싶은 느낌이 크지 않은가... 우선 원래 사용법 링크는 [여기](https://github.com/Platane/snk) 아니면 [여기](https://dev.to/mishmanners/how-to-enable-github-actions-on-your-profile-readme-for-a-contribution-graph-4l66)인데, 둘다 무시하고 걍 혼자했다. 이유는 모르겠는데 계속 오류나고 안되는 것 같아서 걍 맨땅에 헤딩하면서 익혀보았다. 원래 사용법 올라와있는 곳에서는 ```.gif```를 임베딩하는 형태로 자꾸 제시하는데, 실제로 yml 생성해보고 코드 붙여넣으면 그렇지가 않았다. 우선 깃허브 프로필 repository에 들어간다. 그런 뒤 상단에 보이는 메뉴에서 "Actions"를 누른다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/54f3cd64-7d6c-4f9a-aed0-ffac82eb50bf" width="600">
</p>


그런 뒤 <U>"set up a workflow yourself"</U>를 누른다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/8642420e-e335-4383-a067-8fdcf2b7c082" width="1000">
</p>

누르게 되면 ```.github/workflows/main.yml```을 생성하는 화면이 나온다. 여기에 다음과 같은 코드를 작성한 뒤에..

```yml
name: generate animation
on:
  schedule:
    - cron: "0 */24 * * *" 
  workflow_dispatch:
  push:
    branches:
    - master
    
jobs:
  generate:
    permissions: 
      contents: write
    runs-on: ubuntu-latest
    timeout-minutes: 5
    
    steps:
      # generates a snake game from a github user (<github_user_name>) contributions graph, output a svg animation at <svg_out_path>
      - name: generate github-contribution-grid-snake.svg
        uses: Platane/snk/svg-only@v3
        with:
          github_user_name: 여기다가 github ID 적기 !!!
          outputs: |
            dist/github-snake.svg
            dist/github-snake-dark.svg?palette=github-dark

      - name: push github-contribution-grid-snake.svg to the output branch
        uses: crazy-max/ghaction-github-pages@v3.1.0
        with:
          target_branch: output
          build_dir: dist
        env:
          GITHUB_TOKEN: 여기다가는 토큰 적기
```

코드 중간에 잘 보면 <U>"여기다가 github ID 적기 !!!"</U>라고 써있다. 빼먹지 말고 자기 깃허브 아이디 적기..
그리고 <U>토큰을 적는 곳</U>이 있는데 직접 토큰을 구해서 적어도 되고 &#36;&#123;{ secrets.GITHUB_TOKEN }&#125;를 적으면 된다. 완료되었다면 commit을 한다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/a21568e7-9e46-4a59-a2b5-6d259d6b67bd" width="1000">
</p>

커밋하고 나면 main branch로 넘어간 뒤, 다시 Actions 탭으로 들어간다. 그렇게 되면 방금 생성한 ```generate animation``` 탭이 생겨있는 것을 볼 수 있다. 이걸 눌러보자.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/6a8759bb-a895-45f9-b415-b07bcb9f180e" width="1000">
</p>

본인은 이미 workflow를 돌리고 있어서 뜨는 중인데 아마 commite하자마자 들어가면 아무것도 안떠있을 것이다. 그럼 다음과 같이 우측에 있는 Run workflow를 눌러본다. 누른 직후에는 바로 workflow가 안뜨는데, 조금만 기다리면 노란색 build 중인 워크플로우가 뜨게 되고, 이것도 좀 더 인내심을 가지고 쭉 기다리면 빌드가 완료되었다는 표시(파란색 체크)가 나오게 된다.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/530c34bc-050a-4a9c-810f-5e0b6b1600e7" width="300">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/9c459e99-6ec3-4752-892c-e3a9f9584d8b" width="700">
</p>

```yml
with:
    github_user_name: 여기다가 github ID 적기 !!!
    outputs: |
        dist/github-snake.svg
        dist/github-snake-dark.svg?palette=github-dark
```

그리고 아까 작성했던 코드에서 outputs를 보면 두 개가 있는데, 이 중 위에 있는 녀석(```dist/github-snake.svg```)은 light theme에 적용될 snake이고 밑에 있는 녀석(```dist/github-snake-dark.svg?palette=github-dark```)은 dark theme에 적용될 snake이다. 본인은 참고로 다크 테마여서 ```?palette=github-dark```를 적용했다. 이 친구를 깃허브 리드미에 넣고 싶다면 readme.md에서 원하는 위치에

```html
<img src="https://github.com/깃허브ID/깃허브ID/blob/output/github-snake-dark.svg" width="100%">
```

를 넣으면 되고 마찬가지로 깃허브ID를 넣어서 불러올 svg 파일 위치를 소스 url로 명시해주면 된다. 근데 이렇게만 하면 그냥 일반적인 테마의 contribution map이랑 보라색 뱀이 나오는데 본인은 테마 색을 지정해주면서 새로 만들었다. 본인처럼 보라색 contribution map과 파란색 뱀이 쓰고싶은 분들은 앞서 작성했던 ```main.yml``` 코드 중 ```dist/github-snake-dark.svg?palette=github-dark``` 요부분을

```yml
    dist/github-snake-dark.svg?color_snake=#58A6FF&color_dots=#EEEEEE,#E1BEE7,#BA68C8,#8E24AA,#4A148C?palette=github-dark
```

이걸로 대체해주면 된다. color_snake나 color_dots를 매뉴얼하게 바꿔줄수도 있음. ```color_dots```에는 총 5가지의 색상이 들어가야하고 좌측이 contribution 가장 낮은 정도부터 해서 우측이 가장 contribution 높은 색상이라고 보면 된다. 아래처럼 깃허브 잔디 순서처럼 생각하면 됨.

<p align="center">
    <img src="https://github.com/junia3/junia3.github.io/assets/79881119/4e5d1a2f-1b57-4455-b7c7-5ce9f2d55318" width="1000">
</p>

물론 메인 브랜치에서 ```main.yml``` 코드를 바꿨다면 마찬가지로 master 브랜치에서의 ```main.yml``` 코드에도 동일하게 적용해야 제대로된 output이 나오고, 가장 중요한 건 Action에 들어가서 앞서 했던 내용을 그대로 반복하면 된다. 그냥 코드만 바꾸면 아마 안바뀌는 걸로 알고있다. 귀찮으면 그냥 뱀 쓰는것도 추천..

### 마지막 부분에 hit counter 넣기
이 부분은 hit count를 수행할 각자의 주소를 [해당 사이트](https://hits.seeyoufarm.com/)에 넣고 색상이나 로고 등을 지정해주면 된다.
