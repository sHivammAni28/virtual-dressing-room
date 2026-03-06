import argparse
import math
import os
import sys

import cv2
import cvzone
from cvzone.PoseModule import PoseDetector


def calculate_size_recommendation(shoulder_dist: float) -> str:
    """Calculate size recommendation based on shoulder distance in pixels."""
    if shoulder_dist < 80:
        return "XS"
    if shoulder_dist < 100:
        return "S"
    if shoulder_dist < 120:
        return "M"
    if shoulder_dist < 140:
        return "L"
    return "XL"


def calculate_confidence_score(lm_list, shoulder_dist: float) -> float:
    """Calculate a confidence score for the overlay."""
    if not lm_list or len(lm_list) < 25:
        return 0.0

    key_landmarks = [11, 12, 23, 24]
    visible_count = 0

    for idx in key_landmarks:
        if idx < len(lm_list) and len(lm_list[idx]) >= 3:
            if len(lm_list[idx]) > 3 and lm_list[idx][3] > 0.5:
                visible_count += 1
            elif len(lm_list[idx]) == 3:
                visible_count += 1

    visibility_score = visible_count / len(key_landmarks)
    shoulder_stability = min(1.0, shoulder_dist / 50.0) if shoulder_dist > 0 else 0.0

    confidence = (visibility_score * 0.7 + shoulder_stability * 0.3) * 100
    return min(100.0, confidence)


def draw_info_panel(img, size_rec: str, confidence: float, gender: str, shirt_index: int):
    """Render size and confidence panel in the video feed."""
    h, w = img.shape[:2]

    panel_width = 280
    panel_height = 120
    panel_x = w - panel_width - 20
    panel_y = h - panel_height - 20

    overlay = img.copy()
    cv2.rectangle(
        overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    title_text = f"{gender.upper()} SHIRT {shirt_index}"
    cv2.putText(img, title_text, (panel_x + 10, panel_y + 25), font, 0.5, (255, 255, 255), 1)

    size_text = f"Recommended Size: {size_rec}"
    cv2.putText(img, size_text, (panel_x + 10, panel_y + 50), font, font_scale, (0, 255, 0), thickness)

    conf_text = f"Accuracy: {confidence:.1f}%"
    if confidence >= 80:
        conf_color = (0, 255, 0)
    elif confidence >= 60:
        conf_color = (0, 255, 255)
    else:
        conf_color = (0, 0, 255)
    cv2.putText(img, conf_text, (panel_x + 10, panel_y + 75), font, font_scale, conf_color, thickness)

    bar_x = panel_x + 10
    bar_y = panel_y + 90
    bar_width = 200
    bar_height = 15

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    fill_width = int((confidence / 100.0) * bar_width)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), conf_color, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

    return img


def load_shirts(static_dir: str):
    """Load male and female shirt file names."""
    try:
        male_shirts = [f"male/{f}" for f in os.listdir(os.path.join(static_dir, "male")) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        female_shirts = [f"female/{f}" for f in os.listdir(os.path.join(static_dir, "female")) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise RuntimeError(f"Could not load shirt images: {exc}") from exc
    return male_shirts, female_shirts


def select_shirt(gender: str, index: int, male_shirts, female_shirts):
    """Return selected shirt path relative to static directory."""
    if gender == "male":
        if 1 <= index <= len(male_shirts):
            return male_shirts[index - 1]
        raise ValueError(f"Invalid male shirt index: {index} (available: 1-{len(male_shirts)})")
    if gender == "female":
        if 1 <= index <= len(female_shirts):
            return female_shirts[index - 1]
        raise ValueError(f"Invalid female shirt index: {index} (available: 1-{len(female_shirts)})")
    raise ValueError(f"Invalid gender: {gender}")


def run_virtual_tryon(gender: str, shirt_index: int, camera_index: int = 0):
    """Execute the virtual try-on session."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    male_shirts, female_shirts = load_shirts(static_dir)
    selected_shirt = select_shirt(gender, shirt_index, male_shirts, female_shirts)

    shirt_ratio = 581 / 440

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    detector = PoseDetector()
    prev_cx = prev_cy = prev_w = prev_h = 0
    smoothing = 0.2
    size_recommendation = "M"
    confidence_score = 0.0

    print(f"[INFO] Starting Virtual Try-On for {gender} shirt {shirt_index}")
    print(f"[INFO] Selected shirt: {selected_shirt}")
    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("[ERROR] Could not read frame from camera")
                break

            img = detector.findPose(img, draw=False)
            lm_list, _ = detector.findPosition(img, bboxWithHands=False, draw=False)

            confidence_score = calculate_confidence_score(lm_list, 0)

            if lm_list:
                try:
                    lm11, lm12 = lm_list[11], lm_list[12]
                    lm23, lm24 = lm_list[23], lm_list[24]

                    shoulder_dist = math.hypot(lm12[0] - lm11[0], lm12[1] - lm11[1])
                    confidence_score = calculate_confidence_score(lm_list, shoulder_dist)
                    size_recommendation = calculate_size_recommendation(shoulder_dist)

                    if shoulder_dist >= 50:
                        width = int(shoulder_dist * 1.6)
                        height = int(width * shirt_ratio)

                        cx = (lm11[0] + lm12[0]) // 2
                        cy = (lm11[1] + lm12[1]) // 2
                        torso_y = (lm23[1] + lm24[1]) // 2
                        cy_torso = (cy + torso_y) // 2

                        cx = int(prev_cx + (cx - prev_cx) * smoothing)
                        cy_torso = int(prev_cy + (cy_torso - prev_cy) * smoothing)
                        width = int(prev_w + (width - prev_w) * smoothing)
                        height = int(prev_h + (height - prev_h) * smoothing)
                        prev_cx, prev_cy, prev_w, prev_h = cx, cy_torso, width, height

                        shirt_path = os.path.join(static_dir, selected_shirt)
                        if os.path.exists(shirt_path):
                            img_shirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
                            if img_shirt is not None:
                                img_shirt = cv2.resize(img_shirt, (width, height))
                                img_shirt = cv2.flip(img_shirt, 1)

                                dx = lm12[0] - lm11[0]
                                dy = lm12[1] - lm11[1]
                                angle = math.degrees(math.atan2(dy, dx))
                                if dx < 0:
                                    angle += 180

                                matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
                                img_shirt = cv2.warpAffine(
                                    img_shirt, matrix, (width, height), borderMode=cv2.BORDER_TRANSPARENT
                                )

                                x = max(cx - width // 2, 0)
                                y = max(cy_torso - height // 2, 0)
                                img = cvzone.overlayPNG(img, img_shirt, [x, y])
                except Exception as exc:  # pragma: no cover - pose errors
                    print(f"[WARN] Shirt overlay skipped: {exc}")

            img = draw_info_panel(img, size_recommendation, confidence_score, gender, shirt_index)
            cv2.imshow("Virtual Try-On", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Virtual Try-On stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the virtual try-on service")
    parser.add_argument("--gender", choices=["male", "female"], default="male")
    parser.add_argument("--shirt-index", type=int, default=1, dest="shirt_index")
    parser.add_argument("--camera-index", type=int, default=0, dest="camera_index")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_virtual_tryon(args.gender, args.shirt_index, args.camera_index)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
