# Threat Detection System

실시간 카메라 영상에서 사람을 검출하고, 대상의 움직임과 자세를 분석하여 위험도를 시각적으로 표시하는 AR 스타일 HUD 시스템입니다.

YOLOv8을 이용한 사람 검출과 MediaPipe Pose를 이용한 자세 추정을 결합하여 대상의 상태를 분석하며, 위험도에 따라 시스템 상태를 표시합니다.

---

## Features

### Person Detection
- YOLOv8 기반 실시간 사람 검출
- 화면 중앙에 위치한 대상을 자동 선택
- 대상 유지(Target Tracking)

### Motion Analysis
- APPROACHING
- STABLE
- LEAVING

### Pose Analysis
- NORMAL
- ARMS_RAISED
- GUARD_POSE
- PUNCH_LIKE_MOTION

### Threat Assessment
- 위험도 점수 계산
- NORMAL
- ATTENTION
- THREAT

### System State
- NORMAL
- CONDITION
- ENGAGE

---

## System Pipeline

```text
Camera Input
        ↓
YOLOv8 Person Detection
        ↓
Target Selection
        ↓
Motion Analysis
        ↓
MediaPipe Pose Estimation
        ↓
Threat Assessment
        ↓
HUD Visualization
```

---

## Technologies

- Python
- OpenCV
- YOLOv8
- MediaPipe
- NumPy

---

## Demonstration

### NORMAL

대상이 특별한 행동을 하지 않는 일반 상태

![NORMAL](normalstate.png)

---

### CONDITION

가드 자세 또는 주의가 필요한 상태

![CONDITION](guardpose.png)

---

### ENGAGE

펀치 유사 동작이 감지되어 교전 상태로 전환된 상태

![ENGAGE](punchmotion.png)

---

## HUD Information

### Target HUD

```text
TARGET / STABLE
POSE / NORMAL
RISK / NORMAL 15%
```

표시 정보:
- 대상 움직임 상태
- 자세 분석 결과
- 위험도 평가

### System HUD

```text
SYSTEM / NORMAL
SYSTEM / CONDITION
SYSTEM / ENGAGE
```

표시 정보:
- 현재 시스템 대응 상태
- 위협 상황에 따른 상태 전환

---

## Limitations

- 단일 카메라 기반으로 동작하므로 깊이 정보가 제한적입니다.
- 규칙 기반 위험도 분석을 사용하므로 일부 오탐(False Positive)이 발생할 수 있습니다.
- PUNCH_LIKE_MOTION은 손목 이동량을 기반으로 추정하므로 특정 동작이 공격으로 잘못 분류될 수 있습니다.
- 다수 인원 환경에서는 가장 중심에 위치한 인물을 우선적으로 분석합니다.

---

## Future Work

- 위험 행동 분류 모델 적용
- 다중 대상 위험도 비교
- 모바일 AR 디스플레이 연동
- 이벤트 로그 및 스냅샷 저장 기능
- 실시간 스트리밍 최적화

---

## Repository Description

Real-time threat detection and pose analysis HUD using YOLOv8 and MediaPipe.
