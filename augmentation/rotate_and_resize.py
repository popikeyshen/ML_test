import os
import cv2
import argparse
import numpy as np
from glob import glob

# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "dataset_input",
    nargs="?",          # <-- allow optional input
    default="./cars",   # <-- BASE BATCH ADDED HERE
    help="dataset folder with images + yolo txt"
)

parser.add_argument(
    "-o",
    "--output",
    default="generated_dataset",
    help="output folder"
)

parser.add_argument(
    "--min_size",
    type=int,
    default=12,
    help="minimum bbox size"
)

parser.add_argument(
    "--max_size",
    type=int,
    default=100,
    help="maximum bbox size"
)

parser.add_argument(
    "--copies",
    type=int,
    default=10,
    help="generated copies per image"
)

parser.add_argument(
    "--angles",
    nargs="+",
    type=float,
    default=[0, 45, -45, 90, -90,135,-135,180],
    help="rotation angles"
)

parser.add_argument(
    "--show",
    action="store_true"
)

args = parser.parse_args()

# =========================================================
# SETTINGS
# =========================================================
dataset_input = args.dataset_input
dataset_output = args.output

MIN_SIZE = args.min_size
MAX_SIZE = args.max_size

COPIES = args.copies
ANGLES = args.angles

SHOW = args.show

print(f"[INFO] Using dataset: {dataset_input}")

# =========================================================
# OUTPUT DIRS
# =========================================================
images_out = os.path.join(dataset_output, "images")
labels_out = os.path.join(dataset_output, "labels")

os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# =========================================================
# YOLO HELPERS
# =========================================================
def yolo_to_xyxy(label, W, H):

    cls = label[0]

    xc = float(label[1]) * W
    yc = float(label[2]) * H

    bw = float(label[3]) * W
    bh = float(label[4]) * H

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)

    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, W, H):

    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H

    bw = (x2 - x1) / W
    bh = (y2 - y1) / W

    return [cls, xc, yc, bw, bh]

# =========================================================
# ROTATE IMAGE + BBOX
# =========================================================
def rotate_image_and_bbox(image, bbox, angle):

    H, W = image.shape[:2]

    cx = W // 2
    cy = H // 2

    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (W, H),
        borderMode=cv2.BORDER_REFLECT_101
    )

    x1, y1, x2, y2 = bbox

    points = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1]
    ]).T

    transformed = np.dot(matrix, points).T

    rx1 = int(np.min(transformed[:, 0]))
    ry1 = int(np.min(transformed[:, 1]))
    rx2 = int(np.max(transformed[:, 0]))
    ry2 = int(np.max(transformed[:, 1]))

    rx1 = max(0, min(W - 1, rx1))
    ry1 = max(0, min(H - 1, ry1))
    rx2 = max(0, min(W - 1, rx2))
    ry2 = max(0, min(H - 1, ry2))

    return rotated, [rx1, ry1, rx2, ry2]

# =========================================================
# IMAGE FILES
# =========================================================
image_files = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(glob(os.path.join(dataset_input, ext)))

image_files = sorted(image_files)

print("images:", len(image_files))

# =========================================================
# MAIN
# =========================================================
for image_path in image_files:

    base = os.path.splitext(os.path.basename(image_path))[0]

    label_path = os.path.join(dataset_input, base + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(image_path)
    if image is None:
        continue

    H, W = image.shape[:2]

    with open(label_path) as f:
        labels = f.readlines()

    if len(labels) == 0:
        continue

    label = labels[0].strip().split()

    if len(label) != 5:
        continue

    cls, x1, y1, x2, y2 = yolo_to_xyxy(label, W, H)

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if bbox_w <= 0 or bbox_h <= 0:
        continue

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # =====================================================
    # GENERATE
    # =====================================================
    for i in range(COPIES):

        target_size = np.random.randint(MIN_SIZE, MAX_SIZE + 1)

        if bbox_w > bbox_h:
            scale = target_size / bbox_w
        else:
            scale = target_size / bbox_h

        scaled = cv2.resize(image, None, fx=scale, fy=scale)

        scaled_H, scaled_W = scaled.shape[:2]

        scx = int(cx * scale)
        scy = int(cy * scale)

        canvas = np.zeros_like(image)

        start_x = scx - W // 2
        start_y = scy - H // 2

        end_x = start_x + W
        end_y = start_y + H

        src_x1 = max(0, start_x)
        src_y1 = max(0, start_y)
        src_x2 = min(scaled_W, end_x)
        src_y2 = min(scaled_H, end_y)

        dst_x1 = max(0, -start_x)
        dst_y1 = max(0, -start_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

        nx1 = int(x1 * scale - start_x)
        ny1 = int(y1 * scale - start_y)
        nx2 = int(x2 * scale - start_x)
        ny2 = int(y2 * scale - start_y)

        nx1 = max(0, min(W - 1, nx1))
        ny1 = max(0, min(H - 1, ny1))
        nx2 = max(0, min(W - 1, nx2))
        ny2 = max(0, min(H - 1, ny2))

        if nx2 <= nx1 or ny2 <= ny1:
            continue

        for angle in ANGLES:

            rotated_img, rotated_bbox = rotate_image_and_bbox(
                canvas,
                [nx1, ny1, nx2, ny2],
                angle
            )

            rx1, ry1, rx2, ry2 = rotated_bbox

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            yolo_label = xyxy_to_yolo(cls, rx1, ry1, rx2, ry2, W, H)

            angle_name = str(angle).replace("-", "m")

            out_img = os.path.join(images_out, f"{base}_{i}_rot_{angle_name}.jpg")
            out_lbl = os.path.join(labels_out, f"{base}_{i}_rot_{angle_name}.txt")

            cv2.imwrite(out_img, rotated_img)

            with open(out_lbl, "w") as f:
                f.write(" ".join(map(str, yolo_label)))

            if SHOW:
                preview = rotated_img.copy()
                cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
                cv2.imshow("preview", preview)

                if cv2.waitKey(0) == ord("q"):
                    exit()

cv2.destroyAllWindows()
print("DONE")
