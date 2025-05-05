#importing requiring libraries 
import streamlit as st
import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter, speed_estimation
from extract_cor import *
import tempfile
import os
from PIL import Image


#streamlit app 
st.sidebar.title('AI POWERED POLLUTION ESTIMATOR')
app_mode = st.sidebar.selectbox('Choose the App mode:',
                                ['Video Processing with YOLO', 'Pollution Estimator'])

if app_mode == 'Video Processing with YOLO':
    st.title('Video Processing with YOLO')
    video_file = st.file_uploader("Upload a video...", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        video_path = tfile.name

        # Initialize YOLO model
        model = YOLO("yolov8n.pt")
        names = model.names

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points for object counting
        coordinates_count = get_rectangle_coordinates(video_path)
        region_points_count = coordinates_count

        # Define line points for speed estimation
        line_pts = [(0, int(h / 1.7)), (w, int(h / 1.7))]

        # Initialize object counter and speed estimator
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=False,
                 reg_pts=region_points_count,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)
        speed_obj = speed_estimation.SpeedEstimator()
        speed_obj.set_args(reg_pts=line_pts, names=names, view_img=False)

        # Streamlit image placeholder
        image_placeholder = st.empty()

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            # Perform object tracking for counting and speed estimation
            tracks = model.track(im0, persist=True, show=False)
            im0_count = counter.start_counting(im0.copy(), tracks)
            im0_speed = speed_obj.estimate_speed(im0.copy(), tracks)

            # Combine the results
            combined_frame = cv2.addWeighted(im0_count, 0.5, im0_speed, 0.5, 0)
            
            # Convert BGR to RGB
            combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(combined_frame_rgb)

            # Update the image placeholder
            image_placeholder.image(img)

        cap.release()
        cv2.destroyAllWindows()

        st.success("Video processing completed.")

elif app_mode == 'Pollution Estimator':
    st.subheader("Pollution Estimation Inputs")
    number_of_vehicles = st.number_input('Number of Vehicles', min_value=1, value=100)
    average_speed = st.number_input('Average Speed of Vehicles (km/h)', min_value=1, value=50)
    average_distance = st.number_input('Average Distance per Vehicle (km)', min_value=1.0, value=10.0)
    emission_type = st.selectbox('Type of Emission', ['CO2', 'NO2'])

    if st.button('Estimate Pollution'):
        if emission_type == 'CO2':
            emission_factor = 120  # grams per km for CO2
        elif emission_type == 'NO2':
            emission_factor = 0.05  # grams per km for NO2
        
        total_emissions = number_of_vehicles * average_distance * emission_factor
        st.write(f"Estimated {emission_type} emissions based on inputs: {total_emissions} grams.")
