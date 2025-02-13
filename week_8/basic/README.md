# 8주차 기본 과제

![Image](https://github.com/user-attachments/assets/d6fbc3ef-2117-4cab-8405-a24d2277231f)
![Image](https://github.com/user-attachments/assets/de8e841a-05bb-4661-bf14-323873f32b9c)
![Image](https://github.com/user-attachments/assets/d455ee37-af83-4848-aa24-0719f1df3035)

## 정리
rank 8 Max Alloc: 11.8 GB
rank 128 Max Alloc: 12.6 GB
rank 256 Max Alloc: 13.5 GB

각 랭크 사이에 성능 차이는 확인되었으나 그 편차가 크다고 보이지는 않는 것 같음.
train/loss 또한 일치하는 패턴을 보이고 수치상 차이가 나지만 다이나믹 하지는 않다고 봄
다만 rank 8과 128 사이의 차이가 좀더 컷음
