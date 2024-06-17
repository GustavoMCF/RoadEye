import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from sort_module import Sort  # Certifique-se de que sort_module.py está no mesmo diretório


# Função para detectar objetos usando YOLO
def detectar_objetos(img, net, output_layers, classes, colors, confidence_threshold=0.5, nms_threshold=0.3):
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
            if confidence > confidence_threshold and classes[class_id] == 'car':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    detections = [[x, y, x + w, y + h, confidences[i]] for i, (x, y, w, h) in enumerate(boxes) if i in indexes]

    return detections


# Função para configurar a rede YOLOv4-tiny
def configurar_yolo():
    net = cv2.dnn.readNet('Yolo_Files/yolov4-tiny.weights', 'Yolo_Files/yolov4-tiny.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open('Yolo_Files/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, output_layers, classes, colors


# Função para processar o vídeo
def processar_video(uploaded_file, time_threshold, process_every_n_frames):
    net, output_layers, classes, colors = configurar_yolo()
    tracker = Sort()
    previous_positions = {}
    stationary_time = {}
    stationary_threshold = 4  # Pixels
    frame_number = 0
    stopped_cars = set()
    moving_cars = set()

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    notification_placeholder = st.empty()
    stats_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number % process_every_n_frames != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        start_time = time.time()

        detections = detectar_objetos(frame, net, output_layers, classes, colors)
        tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 5)))

        current_stopped_cars = set()
        current_moving_cars = set()

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                movement = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                if movement < stationary_threshold:
                    if obj_id not in stationary_time:
                        stationary_time[obj_id] = time.time()
                    elapsed_time = time.time() - stationary_time[obj_id]
                    if elapsed_time > time_threshold:
                        current_stopped_cars.add(obj_id)
                        if obj_id not in stopped_cars:
                            stopped_cars.add(obj_id)
                            moving_cars.discard(obj_id)
                            notification_placeholder.write(
                                f"Car {obj_id} está parado há mais de {time_threshold} segundos!")
                        label = f"Car {obj_id} (Parado por {int(elapsed_time)}s)"
                    else:
                        current_moving_cars.add(obj_id)
                        label = f"Car {obj_id} (Em movimento)"
                else:
                    stationary_time.pop(obj_id, None)
                    current_moving_cars.add(obj_id)
                    label = f"Car {obj_id} (Em movimento)"
            else:
                label = f"Car {obj_id}"

            previous_positions[obj_id] = (center_x, center_y)

            color = colors[obj_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        stopped_cars = stopped_cars.intersection(current_stopped_cars).union(current_stopped_cars)
        moving_cars = moving_cars.intersection(current_moving_cars).union(current_moving_cars)

        processing_time = time.time() - start_time
        stframe.image(frame, channels="BGR")

        stats_placeholder.markdown(f"""
        ### Estatísticas
        - Tempo de processamento: {processing_time:.2f} segundos por quadro
        - Total de carros rastreados: {len(tracked_objects)}
        - Carros parados: {len(stopped_cars)}
        - Carros em movimento: {len(moving_cars)}
        """)

    cap.release()


# Interface Streamlit
st.title("Detecção e Rastreamento de Carros em Vídeo")

time_threshold = st.sidebar.slider("Tempo para considerar parado (segundos)", 1, 10, 2)  # Segundos
process_every_n_frames = st.sidebar.slider("Processar a cada N quadros", 1, 10, 3)  # Processar a cada N quadros

uploaded_file = st.file_uploader("Faça o upload do vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    processar_video(uploaded_file, time_threshold, process_every_n_frames)