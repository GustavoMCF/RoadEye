import streamlit as st
import cv2
import numpy as np
import tempfile
from sort_module import Sort  # Certifique-se de que sort_module.py está no mesmo diretório


# Função para detectar objetos usando YOLO
def detect_objects(img, net, output_layers, classes, colors):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'car':  # Filtrar apenas carros
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detections.append([x, y, x + w, y + h, confidences[i]])

    return detections


# Carregar a rede YOLO
net = cv2.dnn.readNet('yolo files/yolov3.weights', 'yolo files/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Carregar nomes das classes
with open('yolo files/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Inicializar o rastreador SORT
tracker = Sort()

# Interface Streamlit
st.title("Detecção e Rastreamento de Carros em Vídeo")

uploaded_file = st.file_uploader("Faça o upload do vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame, net, output_layers, classes, colors)
        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            label = f"Car {obj_id}"
            color = colors[obj_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        stframe.image(frame, channels="BGR")

    cap.release()
