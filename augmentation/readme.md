# This script augments car detection datasets in YOLO format.   
## Цей код агментує машинки в папці cars для нашої таски

This script generates augmented YOLO-format training images by resizing objects to random sizes and optionally rotating them.

## What it does

It was created for our car detection task and generates additional training samples by:

- randomly resizing cars,
- keeping the original image size,
- rotating images and bounding boxes,
