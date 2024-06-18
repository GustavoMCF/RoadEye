import cv2
import numpy as np
import streamlit as st
import os

# Função para configurar a rede YOLOv4-tiny
def yolo():
    
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Yolo_Files', 'yolov4-tiny.weights')
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Yolo_Files', 'yolov4-tiny.cfg')
    names_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Yolo_Files', 'coco.names')

    net = cv2.dnn.readNet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    return net, output_layers, classes, colors

def streamlit():
    # Interface Streamlit
    st.title("Detecção e Rastreamento de Carros em Vídeo")
    time_threshold = st.sidebar.slider("Tempo para considerar parado (segundos)", 1, 10, 2)
    process_every_n_frames = st.sidebar.slider("Processar a cada N quadros", 1, 10, 3)
    uploaded_file = st.file_uploader("Faça o upload do vídeo", type=["mp4", "avi", "mov"])
    
    return uploaded_file, time_threshold, process_every_n_frames

