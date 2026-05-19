import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # 가장 가벼운 YOLO 모델
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            break

        results = model(frame, verbose=False)

        persons = []

        height, width = frame.shape[:2]
        center_x = width // 2

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

                if area < 20000:
                    continue

                person_center_x = (x1 + x2) // 2
                dist_from_center = abs(person_center_x - center_x)

                score = area - dist_from_center * 5

                persons.append({
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "area": area,
                    "score": score
                })

        # 타깃 선정
        target_index = None

        if len(persons) > 0:
            target_index = max(range(len(persons)), key=lambda i: persons[i]["score"])

        # 가까운 사람 상위 N명 선정
        MAX_MOTION_PERSONS = 3

        near_indices = sorted(
            range(len(persons)),
            key=lambda i: persons[i]["area"],
            reverse=True
        )[:MAX_MOTION_PERSONS]

        # 화면 그리기
        for i, person in enumerate(persons):

            x1, y1, x2, y2 = person["box"]

            if i == target_index:
                color = (0, 0, 255)
                label = "TARGET"
                thickness = 3

            elif i in near_indices:
                color = (0, 255, 255)
                label = "NEAR"
                thickness = 2

            else:
                color = (0, 255, 0)
                label = "PERSON"
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Threat Detection System", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()