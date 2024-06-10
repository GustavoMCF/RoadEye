import streamlit as st
import cv2
import numpy as np
import tempfile
from sort_module import Sort  # Certifique-se de que sort_module.py está no mesmo diretório

# Função para detectar objetos usando YOLO
def detect_objects(img, net, output_layers, classes, colors, confidence_threshold=0.3):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)  # Aumentar a resolução
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
            if confidence > confidence_threshold and classes[class_id] == 'car':  # Ajustar limiar de confiança
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.3)  # Ajustar NMS
    detections = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detections.append([x, y, x + w, y + h, confidences[i]])

    return detections

# Carregar a rede YOLOv4-tiny
net = cv2.dnn.readNet('Yolo_Files/yolov4-tiny.weights', 'Yolo_Files/yolov4-tiny.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Carregar nomes das classes
with open('Yolo_Files/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Inicializar o rastreador SORT
tracker = Sort()
previous_positions = {}

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
        if len(detections) > 0:
            tracked_objects = tracker.update(np.array(detections))
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                movement = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                if movement < 2:  # Threshold para considerar o carro parado
                    label = f"Car {obj_id} (Parado)"
                else:
                    label = f"Car {obj_id} (Em movimento)"
            else:
                label = f"Car {obj_id}"

            previous_positions[obj_id] = (center_x, center_y)

            color = colors[obj_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        stframe.image(frame, channels="BGR")

    cap.release()
