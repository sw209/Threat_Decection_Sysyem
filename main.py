import cv2
import time
import math
import mediapipe as mp
from ultralytics import YOLO


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def draw_transparent_rect(frame, pt1, pt2, color, alpha=0.35, thickness=-1):
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_transparent_line(frame, pt1, pt2, color, alpha=0.55, thickness=2):
    overlay = frame.copy()
    cv2.line(overlay, pt1, pt2, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_transparent_circle(frame, center, radius, color, alpha=0.65):
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


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


def detect_punch_like_motion(wrist_history):
    if len(wrist_history) < 6:
        return False

    old_left, old_right = wrist_history[0]
    new_left, new_right = wrist_history[-1]

    left_move = math.sqrt((new_left[0] - old_left[0]) ** 2 + (new_left[1] - old_left[1]) ** 2)
    right_move = math.sqrt((new_right[0] - old_right[0]) ** 2 + (new_right[1] - old_right[1]) ** 2)

    return max(left_move, right_move) > 0.35


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


def estimate_system_state(target_motion, pose_state, response_active):
    if response_active:
        return "ENGAGE"

    if target_motion == "APPROACHING":
        return "CONDITION"

    if pose_state in ["ARMS_RAISED", "GUARD_POSE"]:
        return "CONDITION"

    return "NORMAL"


def main():
    model = YOLO("yolov8n.pt")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8
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

    SKELETON_LINE_COLOR = (255, 255, 0)
    SKELETON_DOT_COLOR = (0, 165, 255)
    BODY_BOX_COLOR = (255, 0, 255)
    TEXT_COLOR = (0, 0, 0)
    CENTER_LINE_COLOR = (255, 255, 255)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    target_area_history = []
    motion_history = []
    pose_history = []
    wrist_history = []

    prev_target_center = None
    smoothed_target_box = None
    smoothed_target_body_box = None

    punch_alert_until = 0
    PUNCH_ALERT_DURATION = 1.5

    response_state_until = 0
    RESPONSE_STATE_DURATION = 3.0

    leaving_start_time = None
    LEAVING_RELEASE_DURATION = 2.5

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

        if prev_target_center is not None:
            prev_cx, prev_cy = prev_target_center

            for person in persons:
                bx1, by1, bx2, by2 = person["body_box"]
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2

                d = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

                if d < CENTER_KEEP_THRESHOLD:
                    person["score"] += TARGET_KEEP_BONUS

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
            response_state_until = 0
            leaving_start_time = None

        near_indices = sorted(
            range(len(persons)),
            key=lambda i: persons[i]["area"],
            reverse=True
        )[:MAX_MOTION_PERSONS]

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
                thickness = 2

            elif i in near_indices:
                color = (0, 255, 255)
                label = "NEAR"
                thickness = 1

            else:
                color = (0, 255, 0)
                label = "PERSON"
                thickness = 1

            draw_transparent_rect(frame, (x1, y1), (x2, y2), color, alpha=0.25, thickness=thickness)
            draw_transparent_rect(frame, (bx1, by1), (bx2, by2), BODY_BOX_COLOR, alpha=0.28, thickness=1)

            if i == target_index:
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

                        raw_pose_state = analyze_pose_state(landmarks)

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

                        now = time.time()

                        punch_detected = (
                            raw_pose_state in ["GUARD_POSE", "ARMS_RAISED"]
                            and detect_punch_like_motion(wrist_history)
                        )

                        if punch_detected:
                            punch_alert_until = now + PUNCH_ALERT_DURATION
                            response_state_until = now + RESPONSE_STATE_DURATION

                        if now < punch_alert_until:
                            raw_pose_state = "PUNCH_LIKE_MOTION"

                        if target_motion == "LEAVING":
                            if leaving_start_time is None:
                                leaving_start_time = now
                            elif now - leaving_start_time >= LEAVING_RELEASE_DURATION:
                                response_state_until = 0
                                leaving_start_time = None
                        else:
                            leaving_start_time = None

                        pose_history.append(raw_pose_state)

                        if len(pose_history) > POSE_HISTORY_SIZE:
                            pose_history.pop(0)

                        target_pose_state = max(set(pose_history), key=pose_history.count)

                        target_risk_label, target_risk_score = estimate_threat_score(
                            target_motion,
                            target_pose_state
                        )

                        for start_idx, end_idx in BODY_CONNECTIONS:
                            start = landmarks[start_idx]
                            end = landmarks[end_idx]

                            sx = int(roi_x1 + start.x * roi_w)
                            sy = int(roi_y1 + start.y * roi_h)
                            ex = int(roi_x1 + end.x * roi_w)
                            ey = int(roi_y1 + end.y * roi_h)

                            draw_transparent_line(
                                frame,
                                (sx, sy),
                                (ex, ey),
                                SKELETON_LINE_COLOR,
                                alpha=0.48,
                                thickness=2
                            )

                        for idx in BODY_LANDMARKS:
                            lm = landmarks[idx]
                            px = int(roi_x1 + lm.x * roi_w)
                            py = int(roi_y1 + lm.y * roi_h)

                            draw_transparent_circle(
                                frame,
                                (px, py),
                                4,
                                SKELETON_DOT_COLOR,
                                alpha=0.55
                            )

                draw_transparent_rect(
                    frame,
                    (x1, max(y1 - 85, 0)),
                    (x1 + 380, y1),
                    color,
                    alpha=0.45,
                    thickness=-1
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

        response_active = time.time() < response_state_until

        system_state = estimate_system_state(
            target_motion,
            target_pose_state,
            response_active
        )

        if system_state == "ENGAGE":
            system_color = (0, 0, 255)
        elif system_state == "CONDITION":
            system_color = (0, 255, 255)
        else:
            system_color = (0, 255, 0)

        draw_transparent_rect(
            frame,
            (20, 60),
            (360, 100),
            system_color,
            alpha=0.45,
            thickness=-1
        )

        cv2.putText(
            frame,
            f"SYSTEM / {system_state}",
            (30, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2
        )

        if system_state == "ENGAGE":
            draw_transparent_rect(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), alpha=0.35, thickness=8)

        draw_transparent_line(
            frame,
            (center_x, 0),
            (center_x, height),
            CENTER_LINE_COLOR,
            alpha=0.35,
            thickness=1
        )

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