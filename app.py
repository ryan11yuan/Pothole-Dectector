import os
import uuid
import cv2
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO

app = Flask(__name__)

# Setup directories
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

model = YOLO('best.pt')
names = model.names

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = 1020, 600
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (width, height))
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                label = names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1 + 3, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        if file and file.filename.endswith(('.mp4', '.avi', '.mov')):
            video_id = str(uuid.uuid4())
            filename = f"{video_id}.mp4"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

            file.save(input_path)
            process_video(input_path, output_path)

            video_url = url_for('static', filename=f'processed/{filename}')
            return render_template('index.html', video_url=video_url)
        else:
            return "Invalid file type", 400

    return render_template('index.html', video_url=None)

if __name__ == '__main__':
    app.run(debug=True)
