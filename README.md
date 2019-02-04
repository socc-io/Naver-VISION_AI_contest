# Requirements

    Delf 모델이 tensorflow의 nets를 사용하기 때문에 nets 를 설치해야 합니다.
    https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md


# How to run code

### Debugging

    python dual_main.py --debug --batch_size=32 --dev_percent=0.05 --train_logits --train_sim --stop_gradient_sim --pretrained_model={MY-PATH}/resnet_v1_50.ckpt

### Training on NSML

    nsml run -d ir_ph1_v2 -e dual_main.py --args "--batch_size=128 --dev_percent=0.05 --train_logits --train_sim --stop_gradient_sim --pretrained_model=pre_trained/resnet_v1_50.ckpt"



## 일정
<table class="tbl_schedule">
  <tr>
    <th style="text-align:left;width:50%">일정</th>
    <th style="text-align:center;width:15%">기간</th>
    <th style="text-align:left;width:35%">장소</th>
  </tr>
  <tr>
    <td>
      <strong>참가 신청</strong><br>
      ~2018년 12월 30일(일)
    </td>
    <td style="text-align:center">약 2주</td>
    <td>
      접수 마감
    </td>
  </tr>
  <tr>
    <td>
      <strong>예선 1라운드</strong><br>
      2019년 1월 2일(수) ~ 1월 16일(수) 23:59:59
    </td>
    <td style="text-align:center">약 2주</td>
    <td>
      온라인<br>
      <a href="https://hack.nsml.navercorp.com">https://hack.nsml.navercorp.com</a>
    </td>
  </tr>
  <tr>
    <td>
      <strong>예선 2라운드</strong><br>
      2019년 1월 23일(수) 14:00 ~ 2월 8일(금) 16:00
    </td>
    <td style="text-align:center"> 약 16일</td>
    <td>
      온라인<br>
      <a href="https://hack.nsml.navercorp.com">https://hack.nsml.navercorp.com</a>
    </td>
  </tr>
  <tr>
    <td>
      <strong>결선(온라인)</strong><br>
      2019년 2월 12일(화) 14:00 ~ 2월 20일(수)
    </td>
    <td style="text-align:center">약 9일</td>
    <td>
      온라인<br>
      <a href="https://hack.nsml.navercorp.com">https://hack.nsml.navercorp.com</a>
    </td>
  </tr>
  <tr>
    <td>
      <strong>결선(오프라인)</strong><br>
      2019년 2월 21일(목) ~ 2월 22일(금)
    </td>
    <td style="text-align:center">1박 2일</td>
    <td>
      네이버 커넥트원(춘천)<br>
    </td>
  </tr>
</table>

> ※ 예선 및 결선 참가자에게는 개별로 참가 안내드립니다.<br>
> &nbsp;&nbsp;&nbsp;결선 참가자는 네이버 본사(그린팩토리, 분당)에 모여서 커넥트원(춘천)으로 함께 이동하며<br>
&nbsp;&nbsp;&nbsp;네이버 본사 - 커넥트원 간 이동 차량 및 결선 기간 중 숙식, 간식 등을 제공합니다.

## 미션
* 예선 1차 : 소규모의 라인프렌즈 상품 image retrieval
* 예선 2차 / 결선(온라인, 오프라인) : 대규모의 일반 상품 image retrieval
> ※ 모든 미션은 NSML 플랫폼을 사용해 해결합니다.<br>
> &nbsp;&nbsp;&nbsp;NSML을 통해 미션을 해결하는 방법은 이 [튜토리얼](https://n-clair.github.io/vision-docs/)을 참고해 주세요.

### 예선 1차
예선 1차는 소규모의 라인프렌즈 상품 데이터를 이용한 image retrieval challenge 입니다.
Training data를 이용하여 image retrieval model을 학습하고, test시에는 각 query image(질의 이미지)에 대해 reference images(검색 대상 이미지) 중에서 질의 이미지에 나온 상품과 동일한 상품들을 찾아야 합니다.

#### Training data
Training data는 각 class(상품) 폴더 안에 그 상품을 촬영한 이미지들이 존재합니다.
- Class: 1,000
- Total images: 7,104
- Training data 예시: [Data_example_ph1.zip](https://github.com/AiHackathon2018/AI-Vision/files/2720124/Data_example_ph1.zip)
  - 예선 1차 학습 데이터 중 10개의 클래스이며, 각 클래스의 모든 이미지를 포함합니다.

#### Test data
Test data는 query image와 reference image로 나뉘어져 있습니다.
- Query images: 195
- Reference images: 1,127
- Total images: 1,322

#### 데이터셋 구조
예선 1차, 예선 2차, 결선(온라인, 오프라인) 모두 동일합니다.

```
|-- train
      |-- train_data
            |-- 1141  # 상품 ID
                  |-- s0.jpg
                  |-- s1.jpg
                  |-- s2.jpg
                  ...
            |-- 1142 # 상품 ID
                  |-- s0.jpg
                  |-- s1.jpg
                  |-- s2.jpg
                  ...
             ...
|-- test
      |-- test_data
            |-- query # 질의 이미지 폴더
                  |-- s0.jpg
                  |-- s1.jpg
                  |-- s2.jpg
                  ...
            |-- reference # 검색 대상 이미지 폴더
                  |-- s0.jpg
                  |-- s1.jpg
                  |-- s2.jpg
                  ...
            ...
```

> ※ 폴더 이름은 위와 같지만, 파일 이름은 위 예시와 다를 수 있습니다.

### 예선 2차 / 결선(온라인, 오프라인)
예선 2차는 대규모의 일반 상품 image retrieval challenge 입니다.
예선 1차와 같은 방식이지만, 데이터의 종류가 라인프렌즈로 한정되어 있지 않고, 데이터의 개수가 상대적으로 큰 경우입니다.

### 평가지표
- 평가지표는 image retrieval 분야에서 흔히 쓰이는 [mAP(mean average precision)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)을 사용합니다.
- 동점자가 나올 경우에는 Recall@k를 계산하여 순위를 결정할 수 있습니다.
  - Recall@k 참고: [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/abs/1511.06452)
> For the retrieval task, we use the Recall@K metric. Each test image (query) first retrieves K nearest neighbors from the test set and receives score 1 if an image of the same class is retrieved among the K nearest neighbors and 0 otherwise. Recall@K averages this score over all the images.

## Baseline in NSML

### Baseline model 정보
- Deep learning framework: Keras
- Docker 이미지: `nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2`
- Python 3.6
- 평가지표: mAP
- Epoch=100으로 학습한 결과: mAP 0.0116

### NSML

1. 실행법

    - https://hack.nsml.navercorp.com/download 에서 플랫폼에 맞는 nsml을 다운받습니다.

    - `nsml run`명령어를 이용해서 `main.py`를 실행합니다.

        ```bash
        $ nsml run -d ir_ph1_v2 -e main.py
        ```
2. 제출하기

    - 세션의 모델 정보를 확인합니다.
        ```bash
        $ nsml model ls [session]
        ```
    - 확인한 모델로 submit 명령어를 실행합니다.
        ```bash
        $ nsml submit [session] [checkpoint]
        ```

3. [web](https://hack.nsml.navercorp.com/leaderboard/ir_ph1_v2) 에서 점수를 확인할수있습니다.

### Infer 함수

Submit을 하기위해서는 `infer()`함수에서 [[다음](https://oss.navercorp.com/Hackathon/Vision_AIHackathon/blob/master/main.py#L92)]과 같이 return 포맷을 정해줘야합니다.

대략적인 형태는 아래와 같습니다.

```python
[
    (0, ('query_0', ['refer_12', 'refer_3', 'refer_35', 'refer_87', 'refer_152', 'refer_2', ...])),
    (1, ('query_1', ['refer_2', 'refer_25', 'refer_13', 'refer_7', 'refer_64', 'refer_243', ...])),
     ...
]
```

- 최종 return 형태는 list로 반환해야 합니다.
- `(0, ('query_0', ['refer_12', 'refer_3', 'refer_35', 'refer_87', 'refer_152', 'refer_2', ...]))` tuple
  - 위 형태의 tuple의 첫번째 숫자 값(위의 예제에서는 0)은 query 이미지의 번호이며, 평가와는 무관합니다.
- `('query_0', ['refer_12', 'refer_3', 'refer_35', 'refer_87', 'refer_152', 'refer_2', ...])` tuple
  - `query_0`는 query 이미지 `test_data/query/query_0.jpg`에서 확장자를 뺀 파일명입니다.
  - `refer_12`는 reference 이미지 `test_data/reference/refer_12.jpg`에서 확장자를 뺀 파일명입니다.
  - `['refer_12', 'refer_3', 'refer_35', 'refer_87', 'refer_152', 'refer_2', ...]`은 모든 reference 이미지들을 `query_0`와 가까운 순으로 정렬한 list입니다. (검색 결과의 ranking list)



## 진행 방식 및 심사 기준
### 예선

* 예선 참가팀에게는 예선 기간중 매일 시간당 60-120 NSML 크레딧을 지급합니다.
  (누적 최대치는 2,880이며 리소스 상황에 따라 추가지급될 수 있습니다.)
* 팀 참가자일 경우 대표 팀원에게만 지급합니다.
* 사용하지 않는 크레딧은 누적됩니다.

#### ***예선 1라운드***
* 일정 : 2019. 1. 2 ~ 2019. 1. 16
* NSML 리더보드 순위로 2라운드 진출자 선정 (2라운드 진출팀 50팀 선발,순위가 낮으면 자동 컷오프)


#### ***예선 2라운드***
* 일정 : 2019. 1.16 – 2019. 1. 30
* NSML 리더보드 순위로 결선 진출자 선정 (결선 진출자 약 40팀 선발)
* 전체 인원에 따라 결선 진출팀 수에 변동이 있을 수 있습니다.

### 결선
#### 결선 (온라인)
* 일정 : 2019. 2. 12 – 2019. 2. 20
* 온라인 결선 과정은 오프라인 결선 전, 모델을 향상시키기 위함입니다.
* 온라인 결선을 거치더라도 별도의 컷오프 없이 모든 결선 참여팀이 오프라인 결선에 참여할 수 있습니다.

#### 결선 (오프라인)
* 일정 : 2019. 2. 21 – 2019. 2. 22 1박 2일간 춘천 커넥트원에서 진행
* 최종 우승자는 NSML 리더보드 순위(1위, 2위, 3위)로 결정합니다.
* 결선 참가자에게 제공하는 크레딧은 추후 공지 예정입니다.


> ※ 1 NSML 크레딧으로 NSML GPU를 1분 사용할 수 있습니다.<br>
> &nbsp;&nbsp;&nbsp;10 NSML 크레딧 = GPU 1개 * 10분 = GPU 2개 * 5분 사용

> ※ 예선, 결선 진출자는 개별 안내 드립니다.


## 시상 및 혜택
* 결선 진출자에게는 티셔츠 등의 기념품 증정
* 우수 참가자 중 네이버 입사 지원 시 혜택

## FAQ
* 자주 문의하는 내용을 확인해 보세요! [FAQ.md](https://campaign.naver.com/aihackathon2018/)

## 문의
* 해커톤 관련 문의는 [Q&A issue page](https://github.com/AiHackathon2018/AI-Vision/issues)를 통해 할 수 있습니다.<br>
* 관련 문의는 Tag를 달아 코멘트를 남겨주세요.
* Q&A 문의 답변 시간은 월-금 10:00-19:00 입니다.


## License
```
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
