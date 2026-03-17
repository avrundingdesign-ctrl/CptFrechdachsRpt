FROM python:3.11-slim

# System-Abhängigkeiten für OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements zuerst kopieren (Docker Cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Anwendungscode kopieren
COPY V4_Server.py .
COPY V4_Warp_Image_keypoints.py .
COPY V4_YOLODartKoordinates.py .
COPY V4_DartJsonLogic.py .
COPY V4_Extract_DartCenters.py .
COPY V4_SimulateBoardOnWarpedImageKey.py .

# Modelle kopieren
COPY models/Board.pt models/Board.pt
COPY models/Yolo26Darts_70.pt models/Yolo26Darts_70.pt

# Verzeichnisse anlegen
RUN mkdir -p out uploads jsons

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "V4_Server:app"]
