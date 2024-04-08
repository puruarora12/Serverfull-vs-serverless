import os
import io
from google.cloud import storage
import numpy as np
import json
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO 
import time
import torch
import cv2
import torchvision


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"


storage_client = storage.Client()
input_bucket = 'ds756data'
output_bucket = 'ds756predictions'

def save_image_to_gcs(image, bucket, key):
    # Save the image to the specified GCS bucket
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        image_data = buffer.read()
    elif isinstance(image, bytes):
        image_data = image
    else:
        raise ValueError("Invalid image format. Expected PIL.Image or bytes.")

    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(key)
    blob.upload_from_string(image_data, content_type='image/jpeg')

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        

    return output


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y



def load_image_from_gcs(bucket, key):
    # Load the image from the specified GCS bucket
    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(key)
    image_data = blob.download_as_string()
    image = Image.open(io.BytesIO(image_data))
    return image

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2



@app.route("/detect" , method = ["POST"])
def detect():
    # Load the image from the input GCS bucket
    image = load_image_from_gcs(input_bucket, key)
    print("In detect")
    # print(f"image is {image} and key is {key}")
    # Perform object detection using the YOLO model
    img_copy=image.copy()
    # local_weight_file = 'https://storage.cloud.google.com/ds756data/best.pt'
    # weight_file_path = wget.download(local_weight_file)
    # print(weight_file_path)
    print("trying to get wieght")
    weight_file_path = download_weights_from_gcs(input_bucket, "yolov8n.pt", "/tmp/wieghts.pt")
    print("got weight")
    yolo_model = YOLO(weight_file_path)
    print("model ready")
    results = yolo_model(image)
    print("result ready")
    # pred = non_max_suppression(results, 0.4, 0.5)  # Apply NMS

# Process the predictions and draw bounding boxes and labels
    for i, det in enumerate(results):
        if det is not None and len(det):
            det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], image.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{yolo_model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img_copy, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(img_copy, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw bounding boxes and labels on the image
    # annotated_image = yolo_model.draw_boxes(results, image)
    print("image detected, time to output it to bucket")
    # Save the annotated image to the output GCS bucket
    output_key = f'output/{}'
    print(f"saving to otuput bucket and output key is {output_key}")
    save_image_to_gcs(img_copy, output_bucket, output_key)
    print("image saved to output bucket")

    # Generate a public URL for the annotated image
    image_url = f'https://storage.googleapis.com/{output_bucket}/{output_key}'

    return jsonify({'url': image_url})

def download_weights_from_gcs(bucket, key, local_filename):
    # Download the YOLO weights from the specified GCS bucket
    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(key)
    print(blob)
    blob.download_to_filename(local_filename)
    print(local_filename)
    return local_filename

def gcf_handler(event, context):
    print("in gcf handler")
    print(event['bucket'])
    print(event['name'])

    # if 'data' in event:
    #     data = event['data']
    #     if 'bucket' in data and 'name' in data:
    #         bucket_name = data['bucket']
    #         object_key = data['name']

            # Call your Flask app's detect function
    print("about to call detect from gcf handler")
    result = detect(event['bucket'], event['name'])

#             return {
#                 'statusCode': 200,
#                 'body': json.dumps(result)
#             }
#         else:
#             return {
#                 'statusCode': 400,
#                 'body': json.dumps('Invalid event data format')
#             }
#     else:
#         return {
#             'statusCode': 400,
#             'body': json.dumps('Invalid event format')
#         }

if __name__ == '__main__':
    print("inside main function")
    app.run(debug=True, host='0.0.0.0', port=8080)
