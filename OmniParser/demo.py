from ultralytics import YOLO
from PIL import Image
from utils import *

device = "cpu"
BOX_TRESHOLD = 0.03

model = YOLO("weights/icon_detect/best.pt")
model.to(device)

image_path = "demo.png"

image = Image.open(image_path)
image_rgb = image.convert("RGB")
box_overlay_ratio = image.size[0] / 3200
draw_bbox_config = {
    "text_scale": 0.8 * box_overlay_ratio,
    "text_thickness": max(int(2 * box_overlay_ratio), 1),
    "text_padding": max(int(3 * box_overlay_ratio), 1),
    "thickness": max(int(3 * box_overlay_ratio), 1),
}


caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)

ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
    image_path,
    display_img=False,
    output_bb_format="xyxy",
    goal_filtering=None,
    easyocr_args={"paragraph": False, "text_threshold": 0.9},
    use_paddleocr=True,
)
text, ocr_bbox = ocr_bbox_rslt

dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
    image_path,
    model,
    BOX_TRESHOLD=BOX_TRESHOLD,
    output_coord_in_ratio=False,
    ocr_bbox=ocr_bbox,
    draw_bbox_config=draw_bbox_config,
    caption_model_processor=caption_model_processor,
    ocr_text=text,
    use_local_semantics=True,
    iou_threshold=0.1,
    imgsz=640,
)

image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
print(label_coordinates)
print(parsed_content_list)
image.save("demo-parsed.png")

