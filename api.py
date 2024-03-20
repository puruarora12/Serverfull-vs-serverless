from flask import Flask, request, jsonify
import boto3
import os
import tempfile
from run.py import run

app = Flask(__name__)
s3_client = boto3.client('s3')

def upload_file(bucket, key, local_path, public_read=True):
    try:
        extra_args = {'ACL': 'public-read'} if public_read else {}
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        print(f"File uploaded to S3: {bucket}/{key}")
    except Exception as e:
        print(f"Error uploading file: {e}")

@app.route("/train", methods=["POST"])
def train():
    if 'dataset' not in request.files:
        return jsonify({"error": "No dataset file provided"}), 400

    dataset_file = request.files['dataset']
    input_bucket = request.form['input_bucket']
    input_key = request.form['input_key']
    output_bucket = request.form['output_bucket']
    output_key = request.form['output_key']

    if dataset_file.filename == '':
        return jsonify({"error": "No dataset file selected"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, dataset_file.filename)
        dataset_file.save(dataset_path)

        # Upload the dataset to the specified input S3 bucket and key as public readable
        upload_file(input_bucket, input_key, dataset_path, public_read=True)

        trained_model_path = train(dataset_path)
        if trained_model_path is None:
            return jsonify({"error": "Failed to train model"}), 500

        # Upload the trained model to the specified output S3 bucket and key as private
        upload_file(output_bucket, output_key, trained_model_path)

    return jsonify({"success": True}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
