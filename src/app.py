import video as video
import utils.config as config

uploaded_file, time_threshold, process_every_n_frames = config.streamlit()

if uploaded_file is not None:
    video.processar(uploaded_file, time_threshold, process_every_n_frames)