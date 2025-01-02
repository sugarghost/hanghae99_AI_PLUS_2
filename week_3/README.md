# Named Entity Recognition (NER) Fine-Tuning with Pre-trained and Untrained Models

## Q1) 어떤 task를 선택하셨나요?
> Named Entity Recognition (NER)

NER은 텍스트에서 개체명(예: 장소, 인물, 조직 등)을 식별하고 분류하는 작업입니다. 데이터셋으로 Kaggle에서 제공하는 NER 데이터셋을 활용했습니다.

---

## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> **모델 설계**
- **기반 모델**: BERT (`bert-base-uncased`)를 기반으로 하는 `BertForTokenClassification`을 사용했습니다.
- **입력**: 
  - 토큰화된 텍스트의 ID 배열 (`input_ids`): `(batch_size, max_length)`
  - 패딩 마스크 (`attention_mask`): `(batch_size, max_length)`
  - 실제 태그 (`labels`): `(batch_size, max_length)`
- **출력**:
  - 예측 로짓 값 (`logits`): `(batch_size, max_length, num_labels)`
  - 손실 값 (`loss`): 스칼라 값 (CrossEntropyLoss)

---

## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> **활용 모델**
- `bert-base-uncased`: Hugging Face의 Transformers 라이브러리에서 제공하는 사전 학습된 BERT 모델을 사용했습니다.

---

## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> **비교 분석**
1. **Loss Curve**
   - 초기 손실 값에서 차이가 발생하며 Pre-trained 모델이 5 epoch 학습동안 0.18 -> 0.04로 떨어질때 Untrained 모델은 0.4 -> 0.12 로 격차가 유지되었습니다.

![Training Loss per Epoch](https://github.com/user-attachments/assets/88b6ae57-ec11-4637-be08-8764ea97d4cd)


2. **F1 - Score**
   - Pre-trained 모델은 모든 주요 클래스에서 높은 Precision, Recall, F1-Score를 기록했습니다.
   - Untrained 모델은 주요 클래스에서 상대적으로 낮은 성능을 보이고 모델간 초기 격차가 발생했습니다.

![Pre-trained Model F1-Score](https://github.com/user-attachments/assets/4c13c2c7-533f-4c97-bab8-00dd12428441)
![Untrained Model F1-Score](https://github.com/user-attachments/assets/744aad34-56e1-40c2-875e-3af684f7dd7f)
