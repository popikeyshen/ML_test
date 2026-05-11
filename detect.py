import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models import *
from utils.datasets import *
from utils.utils import *


def letterbox(
    image,
    new_shape=416,
    color=(128, 128, 128),
    auto=False,
    scaleup=True,
    interpolation=cv2.INTER_LINEAR,
):
    """
    Resize image with padding while keeping aspect ratio.
    """

    original_h, original_w = image.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    target_h, target_w = new_shape

    scale = min(target_h / original_h, target_w / original_w)

    if not scaleup:
        scale = min(scale, 1.0)

    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h

    if auto:
        pad_w = np.mod(pad_w, 32)
        pad_h = np.mod(pad_h, 32)

    pad_w /= 2
    pad_h /= 2

    if (original_w, original_h) != (resized_w, resized_h):
        image = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=interpolation,
        )

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return image


def preprocess_frame(frame, img_size, device, half=False):
    """
    Convert OpenCV BGR frame to normalized PyTorch tensor.
    """

    image = letterbox(frame, new_shape=img_size)

    image = image[:, :, ::-1]          # BGR to RGB
    image = image.transpose(2, 0, 1)   # HWC to CHW
    image = np.ascontiguousarray(image)

    image = torch.from_numpy(image).to(device)

    if half:
        image = image.half()
    else:
        image = image.float()

    image /= 255.0

    if image.ndimension() == 3:
        image = image.unsqueeze(0)

    return image


def init_model(args):
    """
    Load YOLO/Darknet model.
    """

    device = torch_utils.select_device(
        device="cpu" if ONNX_EXPORT else args.device
    )

    model = Darknet(args.cfg, args.img_size)

    attempt_download(args.weights)

    if args.weights.endswith(".pt"):
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        load_darknet_weights(model, args.weights)

    model.to(device).eval()

    half = args.half and device.type != "cpu"

    if half:
        model.half()

    names = load_classes(args.names)

    colors = [
        [random.randint(0, 255) for _ in range(3)]
        for _ in range(len(names))
    ]

    return device, model, names, colors, half


def detect_frame(frame, model, device, names, colors, args, half):
    """
    Run detection on one video frame.
    """

    input_tensor = preprocess_frame(
        frame=frame,
        img_size=args.img_size,
        device=device,
        half=half,
    )

    prediction = model(input_tensor)[0]

    if half:
        prediction = prediction.float()

    prediction = non_max_suppression(
        prediction,
        args.conf_thres,
        args.nms_thres,
    )

    result_frame = frame.copy()

    for detections in prediction:
        if detections is None or len(detections) == 0:
            continue

        detections[:, :4] = scale_coords(
            input_tensor.shape[2:],
            detections[:, :4],
            result_frame.shape,
        ).round()

        for *xyxy, confidence, class_id in detections:
            class_id = int(class_id)

            # show only class 5
            if class_id != 4:
                continue

            label = f"{names[class_id]} {confidence:.2f}"

            plot_one_box(
                xyxy,
                result_frame,
                label=label,
                color=colors[class_id],
            )

    return result_frame


def open_video_source(source):
    """
    Open video file or webcam.
    """

    if source.isdigit():
        source = int(source)

    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    return video


def create_video_writer(video, output_path, fourcc_name):
    """
    Create output video writer with same size and FPS as input video.
    """

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    return writer, width, height, fps


def detect_video(args):
    """
    Main video inference loop.
    """

    device, model, names, colors, half = init_model(args)

    video = open_video_source(args.source)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output_video_name

    writer, width, height, fps = create_video_writer(
        video=video,
        output_path=output_path,
        fourcc_name=args.fourcc,
    )

    if args.start_frame > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    print(f"[INFO] Source: {args.source}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Video size: {width}x{height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Image size for model: {args.img_size}")
    print(f"[INFO] Half precision: {half}")

    frame_id = 0
    total_start_time = time.time()

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_start_time = time.time()

        result_frame = detect_frame(
            frame=frame,
            model=model,
            device=device,
            names=names,
            colors=colors,
            args=args,
            half=half,
        )

        writer.write(result_frame)

        elapsed = time.time() - frame_start_time

        print(f"[INFO] Frame {frame_id} processed in {elapsed:.3f}s")

        frame_id += 1


        #### RESIZE TO SCREEN SIZE - NOT VIDEO SIZE
        display_frame = result_frame

        h, w = display_frame.shape[:2]

        target_width = 800


        scale = target_width / w

        new_w = int(w * scale)
        new_h = int(h * scale)

        display_frame = cv2.resize(
                display_frame,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
        )
        #### RESIZE TO SCREEN SIZE - NOT VIDEO SIZE

        #cv2.imshow("Detection result", result_frame)
        cv2.imshow("Detection 222 result", display_frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
                break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    total_elapsed = time.time() - total_start_time

    print("[INFO] Done")
    print(f"[INFO] Processed frames: {frame_id}")
    print(f"[INFO] Total time: {total_elapsed:.3f}s")
    print(f"[INFO] Saved video to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default="data/a1.cfg",
        help="path to model cfg file",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="a1.weights",
        help="path to model weights file",
    )

    parser.add_argument(
        "--names",
        type=str,
        default="data/37.names",
        help="path to class names file",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="input.mp4",
        help="input video file or webcam id, for example 0",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="output folder",
    )

    parser.add_argument(
        "--output-video-name",
        type=str,
        default="result.mp4",
        help="output video filename",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=832,
        help="inference image size",
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.9,
        help="object confidence threshold",
    )

    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.4,
        help="IoU threshold for non-maximum suppression",
    )

    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help="use FP16 inference on CUDA",
    )

    parser.add_argument(
        "--device",
        default="",
        help="device id, for example 0, 0,1, or cpu",
    )


    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="start frame for video inference",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    with torch.no_grad():
        detect_video(args)


if __name__ == "__main__":
    main()
