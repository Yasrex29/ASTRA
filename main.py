import cv2
from src.detector import detect_people
from src.heatmap import generate_heatmap
from src.collision import collision_alert

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    boxes, centers = detect_people(frame)

    heatmap_frame = generate_heatmap(frame, centers)

    output = collision_alert(heatmap_frame, centers)

    cv2.putText(
        output,
        f"Crowd Count: {len(centers)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("Crowd Monitoring System", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
