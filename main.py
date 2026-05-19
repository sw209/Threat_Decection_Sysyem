import cv2
import time
import mediapipe as mp
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7
    )

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

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    target_area_history = []
    motion_history = []
    prev_target_center = None

    smoothed_target_box = None
    smoothed_target_body_box = None
    SMOOTHING_ALPHA = 0.7

    MOTION_HISTORY_SIZE = 5
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

                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                box_width = x2 - x1
                box_height = y2 - y1
                area = box_width * box_height

                if area < MIN_PERSON_AREA:
                    continue

                # 얼굴~몸통 중심 영역
                body_x1 = x1 + int(box_width * 0.2)
                body_x2 = x2 - int(box_width * 0.2)
                body_y1 = y1 + int(box_height * 0.15)
                body_y2 = y1 + int(box_height * 0.65)

                body_width = body_x2 - body_x1
                body_height = body_y2 - body_y1
                body_area = body_width * body_height

                body_center_x = (body_x1 + body_x2) // 2
                dist_from_center = abs(body_center_x - center_x)

                score = body_area - dist_from_center * 5

                persons.append({
                    "box": (x1, y1, x2, y2),
                    "body_box": (body_x1, body_y1, body_x2, body_y2),
                    "conf": conf,
                    "area": area,
                    "body_area": body_area,
                    "score": score
                })

        # 2. 이전 타겟 유지 보너스
        if prev_target_center is not None:
            prev_cx, prev_cy = prev_target_center

            for person in persons:
                bx1, by1, bx2, by2 = person["body_box"]
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2

                dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

                if dist < CENTER_KEEP_THRESHOLD:
                    person["score"] += TARGET_KEEP_BONUS

        # 3. 메인 타겟 선정
        target_index = None

        if len(persons) > 0:
            target_index = max(
                range(len(persons)),
                key=lambda i: persons[i]["score"]
            )

            bx1, by1, bx2, by2 = persons[target_index]["body_box"]
            prev_target_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        else:
            prev_target_center = None
            target_area_history.clear()
            motion_history.clear()

        # 4. 가까운 인물 상위 N명
        near_indices = sorted(
            range(len(persons)),
            key=lambda i: persons[i]["area"],
            reverse=True
        )[:MAX_MOTION_PERSONS]

        # 5. 타겟 움직임 분석
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
                    color = (0, 0, 255)
                elif target_motion == "LEAVING":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

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

            # 사람 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 몸통 기준 박스
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

            if i == target_index:
                # HUD 텍스트
                cv2.rectangle(
                    frame,
                    (x1, max(y1 - 35, 0)),
                    (x1 + 330, y1),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # TARGET에게만 skeleton 표시
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

                        for start_idx, end_idx in BODY_CONNECTIONS:
                            start = landmarks[start_idx]
                            end = landmarks[end_idx]

                            sx = int(roi_x1 + start.x * roi_w)
                            sy = int(roi_y1 + start.y * roi_h)
                            ex = int(roi_x1 + end.x * roi_w)
                            ey = int(roi_y1 + end.y * roi_h)

                            cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2)

                        for idx in BODY_LANDMARKS:
                            lm = landmarks[idx]
                            px = int(roi_x1 + lm.x * roi_w)
                            py = int(roi_y1 + lm.y * roi_h)
                            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

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
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)

        # FPS
        current_time = time.time()
        fps = 1 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Threat Detection System", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()