#!/usr/bin/env python3
import os
from datetime import datetime
from google.cloud import storage

# Pfade
UPLOAD_FOLDER = "/opt/dartvision/uploads"
SERVICE_ACCOUNT_KEY = "/opt/dartvision/dart-vision-key.json"
BUCKET_NAME = "dartvision-uploads"

def upload_files_to_gcs():
    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    for filename in os.listdir(UPLOAD_FOLDER):
        local_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(local_path):
            continue
        
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_filename(local_path)
        print(f"✅ Hochgeladen: {filename}")

    print("🎯 Upload abgeschlossen:", datetime.now())

if __name__ == "__main__":
    upload_files_to_gcs()
