import cv2
import time
import math
import mediapipe as mp
from ultralytics import YOLO


# 두 landmark 사이 거리 계산
def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# 정적인 자세 분석
def analyze_pose_state(landmarks):
    nose = landmarks[0]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    shoulder_width = dist(left_shoulder, right_shoulder)

    left_wrist_near_face = dist(left_wrist, nose) < shoulder_width * 0.9
    right_wrist_near_face = dist(right_wrist, nose) < shoulder_width * 0.9

    left_arm_raised = left_wrist.y < left_shoulder.y
    right_arm_raised = right_wrist.y < right_shoulder.y

    left_elbow_bent = dist(left_wrist, left_shoulder) < shoulder_width * 1.1
    right_elbow_bent = dist(right_wrist, right_shoulder) < shoulder_width * 1.1

    if left_wrist_near_face and right_wrist_near_face and (left_elbow_bent or right_elbow_bent):
        return "GUARD_POSE"

    if left_arm_raised or right_arm_raised:
        return "ARMS_RAISED"

    return "NORMAL"


# 손목 이동량 기반 펀치 유사 동작 감지
def detect_punch_like_motion(wrist_history):
    if len(wrist_history) < 6:
        return False

    old_left, old_right = wrist_history[0]
    new_left, new_right = wrist_history[-1]

    left_move = math.sqrt(
        (new_left[0] - old_left[0]) ** 2 +
        (new_left[1] - old_left[1]) ** 2
    )

    right_move = math.sqrt(
        (new_right[0] - old_right[0]) ** 2 +
        (new_right[1] - old_right[1]) ** 2
    )

    # 민감도 조정: 값이 낮을수록 더 쉽게 펀치로 판단
    return max(left_move, right_move) > 0.22


# 움직임 + 자세 기반 위험도 점수 계산
def estimate_threat_score(target_motion, pose_state):
    score = 10

    if target_motion == "APPROACHING":
        score += 25
    elif target_motion == "STABLE":
        score += 5
    elif target_motion == "LEAVING":
        score -= 15

    if pose_state == "ARMS_RAISED":
        score += 20
    elif pose_state == "GUARD_POSE":
        score += 35
    elif pose_state == "PUNCH_LIKE_MOTION":
        score += 45

    if target_motion == "APPROACHING" and pose_state == "GUARD_POSE":
        score += 15

    if target_motion == "APPROACHING" and pose_state == "PUNCH_LIKE_MOTION":
        score += 25

    score = max(0, min(score, 100))

    if score >= 70:
        label = "THREAT"
    elif score >= 40:
        label = "ATTENTION"
    else:
        label = "NORMAL"

    return label, score


def main():
    # YOLO 사람 검출 모델
    model = YOLO("yolov8n.pt")

    # MediaPipe Pose 설정
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8
    )

    # 얼굴 landmark를 제외한 몸통/팔/다리 연결선
    BODY_CONNECTIONS = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]

    BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    # OpenCV 색상은 BGR 순서
    SKELETON_LINE_COLOR = (255, 255, 0)   # 청록
    SKELETON_DOT_COLOR = (0, 165, 255)    # 주황
    BODY_BOX_COLOR = (255, 0, 255)        # 보라
    TEXT_COLOR = (0, 0, 0)                # 검정
    CENTER_LINE_COLOR = (255, 255, 255)   # 흰색

    # 카메라 입력
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    # 상태 기록용 변수
    target_area_history = []
    motion_history = []
    pose_history = []
    wrist_history = []

    prev_target_center = None
    smoothed_target_box = None
    smoothed_target_body_box = None

    # 펀치 감지 후 일정 시간 유지
    punch_alert_until = 0
    PUNCH_ALERT_DURATION = 1.5

    # 설정값
    SMOOTHING_ALPHA = 0.7
    MOTION_HISTORY_SIZE = 5
    POSE_HISTORY_SIZE = 5
    WRIST_HISTORY_SIZE = 8
    MAX_MOTION_PERSONS = 3

    TARGET_KEEP_BONUS = 20000
    CENTER_KEEP_THRESHOLD = 100
    MIN_PERSON_AREA = 20000

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            break

        height, width = frame.shape[:2]
        center_x = width // 2

        results = model(frame, verbose=False)
        persons = []

        # 1. 사람 후보 수집
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])

                # COCO dataset에서 person class id는 0
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                box_width = x2 - x1
                box_height = y2 - y1
                area = box_width * box_height

                # 너무 작은 사람은 제외
                if area < MIN_PERSON_AREA:
                    continue

                # 팔 흔들림 영향을 줄이기 위한 얼굴~몸통 기준 박스
                body_x1 = x1 + int(box_width * 0.2)
                body_x2 = x2 - int(box_width * 0.2)
                body_y1 = y1 + int(box_height * 0.15)
                body_y2 = y1 + int(box_height * 0.65)

                body_width = body_x2 - body_x1
                body_height = body_y2 - body_y1
                body_area = body_width * body_height

                body_center_x = (body_x1 + body_x2) // 2
                dist_from_center = abs(body_center_x - center_x)

                # 중앙에 가깝고 몸통 영역이 큰 사람을 우선 타겟으로 선정
                score = body_area - dist_from_center * 5

                persons.append({
                    "box": (x1, y1, x2, y2),
                    "body_box": (body_x1, body_y1, body_x2, body_y2),
                    "conf": conf,
                    "area": area,
                    "body_area": body_area,
                    "score": score
                })

        # 2. 이전 타겟 근처 후보에게 보너스 부여
        if prev_target_center is not None:
            prev_cx, prev_cy = prev_target_center

            for person in persons:
                bx1, by1, bx2, by2 = person["body_box"]
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2

                d = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

                if d < CENTER_KEEP_THRESHOLD:
                    person["score"] += TARGET_KEEP_BONUS

        # 3. 메인 타겟 선정
        target_index = None

        if len(persons) > 0:
            target_index = max(range(len(persons)), key=lambda i: persons[i]["score"])

            bx1, by1, bx2, by2 = persons[target_index]["body_box"]
            prev_target_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        else:
            prev_target_center = None
            target_area_history.clear()
            motion_history.clear()
            pose_history.clear()
            wrist_history.clear()
            punch_alert_until = 0

        # 4. 가까운 인물 상위 N명 선정
        near_indices = sorted(
            range(len(persons)),
            key=lambda i: persons[i]["area"],
            reverse=True
        )[:MAX_MOTION_PERSONS]

        # 5. 타겟 접근/이탈 분석
        target_motion = "UNKNOWN"

        if target_index is not None:
            current_area = persons[target_index]["area"]
            target_area_history.append(current_area)

            if len(target_area_history) > 10:
                old_area = target_area_history[-10]
                diff_ratio = (current_area - old_area) / max(old_area, 1)

                if diff_ratio > 0.03:
                    target_motion = "APPROACHING"
                elif diff_ratio < -0.03:
                    target_motion = "LEAVING"
                else:
                    target_motion = "STABLE"

                motion_history.append(target_motion)

                if len(motion_history) > MOTION_HISTORY_SIZE:
                    motion_history.pop(0)

                target_motion = max(set(motion_history), key=motion_history.count)

            if len(target_area_history) > 30:
                target_area_history.pop(0)

        # 6. TARGET 박스 smoothing
        display_target_box = None
        display_target_body_box = None

        if target_index is not None:
            current_box = persons[target_index]["box"]
            current_body_box = persons[target_index]["body_box"]

            if smoothed_target_box is None:
                smoothed_target_box = current_box
            else:
                smoothed_target_box = tuple(
                    int(SMOOTHING_ALPHA * smoothed_target_box[j] + (1 - SMOOTHING_ALPHA) * current_box[j])
                    for j in range(4)
                )

            if smoothed_target_body_box is None:
                smoothed_target_body_box = current_body_box
            else:
                smoothed_target_body_box = tuple(
                    int(SMOOTHING_ALPHA * smoothed_target_body_box[j] + (1 - SMOOTHING_ALPHA) * current_body_box[j])
                    for j in range(4)
                )

            display_target_box = smoothed_target_box
            display_target_body_box = smoothed_target_body_box

        else:
            smoothed_target_box = None
            smoothed_target_body_box = None

        target_pose_state = "NO_POSE"
        target_risk_label = "NORMAL"
        target_risk_score = 0

        # 7. 화면 출력
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person["box"]
            bx1, by1, bx2, by2 = person["body_box"]

            if i == target_index:
                if display_target_box is not None:
                    x1, y1, x2, y2 = display_target_box

                if display_target_body_box is not None:
                    bx1, by1, bx2, by2 = display_target_body_box

                if target_motion == "APPROACHING":
                    color = (0, 0, 255)       # 빨강
                elif target_motion == "LEAVING":
                    color = (0, 255, 255)     # 노랑
                else:
                    color = (0, 255, 0)       # 초록

                label = f"TARGET / {target_motion}"
                thickness = 3

            elif i in near_indices:
                color = (0, 255, 255)
                label = "NEAR"
                thickness = 2

            else:
                color = (0, 255, 0)
                label = "PERSON"
                thickness = 1

            # 사람 박스와 몸통 기준 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), BODY_BOX_COLOR, 1)

            if i == target_index:
                # TARGET에 대해서만 Pose 추정
                roi_x1 = max(x1, 0)
                roi_y1 = max(y1, 0)
                roi_x2 = min(x2, width)
                roi_y2 = min(y2, height)

                target_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                if target_roi.size > 0:
                    target_rgb = cv2.cvtColor(target_roi, cv2.COLOR_BGR2RGB)
                    pose_result = pose.process(target_rgb)

                    if pose_result.pose_landmarks:
                        landmarks = pose_result.pose_landmarks.landmark
                        roi_w = roi_x2 - roi_x1
                        roi_h = roi_y2 - roi_y1

                        # 정적 자세 분석
                        raw_pose_state = analyze_pose_state(landmarks)

                        # 손목 이동 기록
                        left_wrist = landmarks[15]
                        right_wrist = landmarks[16]

                        wrist_history.append(
                            (
                                (left_wrist.x, left_wrist.y),
                                (right_wrist.x, right_wrist.y)
                            )
                        )

                        if len(wrist_history) > WRIST_HISTORY_SIZE:
                            wrist_history.pop(0)

                        # 펀치 감지 후 일정 시간 유지
                        now = time.time()

                        if detect_punch_like_motion(wrist_history):
                            punch_alert_until = now + PUNCH_ALERT_DURATION

                        if now < punch_alert_until:
                            raw_pose_state = "PUNCH_LIKE_MOTION"

                        # 자세 상태 smoothing
                        pose_history.append(raw_pose_state)

                        if len(pose_history) > POSE_HISTORY_SIZE:
                            pose_history.pop(0)

                        target_pose_state = max(set(pose_history), key=pose_history.count)

                        # 위험도 계산
                        target_risk_label, target_risk_score = estimate_threat_score(
                            target_motion,
                            target_pose_state
                        )

                        # skeleton 선
                        for start_idx, end_idx in BODY_CONNECTIONS:
                            start = landmarks[start_idx]
                            end = landmarks[end_idx]

                            sx = int(roi_x1 + start.x * roi_w)
                            sy = int(roi_y1 + start.y * roi_h)
                            ex = int(roi_x1 + end.x * roi_w)
                            ey = int(roi_y1 + end.y * roi_h)

                            cv2.line(frame, (sx, sy), (ex, ey), SKELETON_LINE_COLOR, 2)

                        # skeleton 점
                        for idx in BODY_LANDMARKS:
                            lm = landmarks[idx]
                            px = int(roi_x1 + lm.x * roi_w)
                            py = int(roi_y1 + lm.y * roi_h)
                            cv2.circle(frame, (px, py), 4, SKELETON_DOT_COLOR, -1)

                # TARGET HUD
                cv2.rectangle(
                    frame,
                    (x1, max(y1 - 85, 0)),
                    (x1 + 380, y1),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    TEXT_COLOR,
                    2
                )

                cv2.putText(
                    frame,
                    f"POSE / {target_pose_state}",
                    (x1 + 5, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    TEXT_COLOR,
                    2
                )

                cv2.putText(
                    frame,
                    f"RISK / {target_risk_label} {target_risk_score}%",
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    TEXT_COLOR,
                    2
                )

            else:
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # 중앙선
        cv2.line(frame, (center_x, 0), (center_x, height), CENTER_LINE_COLOR, 1)

        # FPS 표시
        current_time = time.time()
        fps = 1 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 출력 화면 확대
        display_frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
        cv2.imshow("Threat Detection System", display_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()