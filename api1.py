import os
import io
import boto3
import awsgi
import numpy as np
import json
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO # Assuming you have a YOLO model file named 'yolo_model.py'

app = Flask(__name__)

s3 = boto3.client('s3')
input_bucket = 'ds756data'
output_bucket = 'ds756predictions'

def save_image_to_s3(image, bucket, key):
    # Save the image to the specified S3 bucket
    """
    Save the image to the specified S3 bucket.

    Args:
        image (PIL.Image or bytes): The image to save.
        bucket (str): The name of the S3 bucket.
        key (str): The unique key for the image in the S3 bucket.
    """
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        image_data = buffer.read()
    elif isinstance(image, bytes):
        image_data = image
    else:
        raise ValueError("Invalid image format. Expected PIL.Image or bytes.")

    s3.put_object(Bucket=bucket, Key=key, Body=image_data, ContentType='image/jpeg')

def load_image_from_s3(bucket, key):
    # Load the image from the specified S3 bucket
    """
    Load the image from the specified S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The unique key for the image in the S3 bucket.

    Returns:
        PIL.Image: The loaded image.
    """
    image_data = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    image = Image.open(io.BytesIO(image_data))
    return image

@app.route('/detect', methods=['POST'])
def detect(bucket, key):
    # file = request.files['image']
    # filename = file.filename

    # Save the image to the input S3 bucket
    # save_image_to_s3(file, input_bucket, filename)

    # Load the image from the input S3 bucket
    image = load_image_from_s3(input_bucket, key)

    # Perform object detection using the YOLO model
    local_weight_file = 'best.pt'
    weight_file_path = download_weights_from_s3(input_bucket, "best.pt", local_weight_file)
    yolo_model= YOLO(weight_file_path)
    results = yolo_model.detect(image)

    # Draw bounding boxes and labels on the image
    annotated_image = yolo_model.draw_boxes(results, image)

    # Save the annotated image to the output S3 bucket
    output_key = f'output/{key}'
    save_image_to_s3(annotated_image, output_bucket, output_key)

    # Generate a public URL for the annotated image
    image_url = f'https://s3.amazonaws.com/{output_bucket}/{output_key}'

    return jsonify({'url': image_url})




def download_weights_from_s3(bucket, key, local_filename):
    """
    Download the YOLO weights from the specified S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The unique key for the weight file in the S3 bucket.
        local_filename (str): The local file path to save the weight file.

    Returns:
        str: The local file path of the downloaded weight file.
    """
    s3.download_file(bucket, key, local_filename)
    return local_filename



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

def lambda_handler(event , context):
    if event['Records']:
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        object_key = s3_event['object']['key']

        # Call your Flask app's detect function
        result = app.detect(bucket_name, object_key)

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid event format')
        }