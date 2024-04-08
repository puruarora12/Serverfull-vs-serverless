import os
import io
from google.cloud import storage
import numpy as np
import json
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO  # Assuming you have a YOLO model file named 'yolo_model.py'

app = Flask(__name__)

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

def load_image_from_gcs(bucket, key):
    # Load the image from the specified GCS bucket
    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(key)
    image_data = blob.download_as_string()
    image = Image.open(io.BytesIO(image_data))
    return image

@app.route('/detect', methods=['POST'])
def detect(bucket, key):
    # Load the image from the input GCS bucket
    image = load_image_from_gcs(input_bucket, key)

    # Perform object detection using the YOLO model
    local_weight_file = 'best.pt'
    weight_file_path = download_weights_from_gcs(input_bucket, "best.pt", local_weight_file)
    yolo_model = YOLO(weight_file_path)
    results = yolo_model.detect(image)

    # Draw bounding boxes and labels on the image
    annotated_image = yolo_model.draw_boxes(results, image)

    # Save the annotated image to the output GCS bucket
    output_key = f'output/{key}'
    save_image_to_gcs(annotated_image, output_bucket, output_key)

    # Generate a public URL for the annotated image
    image_url = f'https://storage.googleapis.com/{output_bucket}/{output_key}'

    return jsonify({'url': image_url})

def download_weights_from_gcs(bucket, key, local_filename):
    # Download the YOLO weights from the specified GCS bucket
    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(key)
    blob.download_to_filename(local_filename)
    return local_filename

def gcf_handler(event, context):
    if 'data' in event:
        data = event['data']
        if 'bucket' in data and 'name' in data:
            bucket_name = data['bucket']
            object_key = data['name']

            # Call your Flask app's detect function
            result = app.detect(bucket_name, object_key)

            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps('Invalid event data format')
            }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid event format')
        }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
