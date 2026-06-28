import cv2
import numpy as np
import plotly.express as px
import streamlit as st
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(page_title="AI Traffic Risk Dashboard", layout="wide")
st.title("🚦 Universal Smart Traffic Monitoring System")
st.markdown("### Computer Vision (YOLOv8) + Fuzzy Logic Risk Assessment + PDC")

distance_in = ctrl.Antecedent(np.arange(0, 501, 1), 'distance')
speed_in = ctrl.Antecedent(np.arange(0, 121, 1), 'speed')
risk_out = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

distance_in['close'] = fuzz.trimf(distance_in.universe, [0, 0, 150])
distance_in['medium'] = fuzz.trimf(distance_in.universe, [100, 200, 350])
distance_in['far'] = fuzz.trapmf(distance_in.universe, [250, 400, 500, 500])

speed_in['slow'] = fuzz.trimf(speed_in.universe, [0, 0, 50])
speed_in['moderate'] = fuzz.trimf(speed_in.universe, [40, 65, 85])
speed_in['fast'] = fuzz.trapmf(speed_in.universe, [75, 100, 120, 120])

risk_out['low'] = fuzz.trimf(risk_out.universe, [0, 0, 45])
risk_out['medium'] = fuzz.trimf(risk_out.universe, [35, 50, 70])
risk_out['high'] = fuzz.trimf(risk_out.universe, [60, 100, 100])

rule1 = ctrl.Rule(distance_in['close'] & speed_in['fast'], risk_out['high'])
rule2 = ctrl.Rule(distance_in['close'] & speed_in['moderate'], risk_out['high'])
rule3 = ctrl.Rule(distance_in['medium'] & speed_in['fast'], risk_out['high'])
rule4 = ctrl.Rule(distance_in['medium'] & speed_in['moderate'], risk_out['medium'])
rule5 = ctrl.Rule(distance_in['far'] | speed_in['slow'], risk_out['low'])

risk_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
risk_simulation = ctrl.ControlSystemSimulation(risk_control)

def get_fuzzy_risk(dist_val, speed_val):
    try:
        risk_simulation.input['distance'] = np.clip(dist_val, 0, 500)
        risk_simulation.input['speed'] = np.clip(speed_val, 0, 120)
        risk_simulation.compute()
        return risk_simulation.output['risk']
    except:
        return 0.0

st.sidebar.header("🔧 Input Settings")
source_type = st.sidebar.radio(
    "Select Traffic Source:",
    ("Upload Video File", "Enter Video URL / IP Stream", "Live Webcam / Real-Time Camera")
)

video_source = None

if source_type == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Traffic Video (.mp4)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    else:
        st.info("👋 Welcome! Please upload a video file from the sidebar to start processing.")

elif source_type == "Enter Video URL / IP Stream":
    url_input = st.sidebar.text_input("Paste Video Link or RTSP IP Cam URL:", placeholder="https://example.com/traffic.mp4")
    if url_input:
        video_source = url_input
    else:
        st.info("🔗 Please paste a valid video URL or network camera link in the sidebar.")

elif source_type == "Live Webcam / Real-Time Camera":
    st.sidebar.warning("Note: The live webcam on cloud deployment will only work when the user allows camera permissions.")
    video_source = 0

if video_source is not None:

    col1, col2, col3, col4 = st.columns(4)
    metric_total = col1.empty()
    metric_avg_speed = col2.empty()
    metric_max_risk = col3.empty()
    metric_status = col4.empty()

    video_col, stats_col = st.columns([2, 1])
    video_frame_holder = video_col.empty()
    chart_holder = stats_col.empty()

    speed_history = []
    risk_history = []
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        st.error("Error: Video source could not be loaded. Please check the link, file, or camera settings.")
    else:
        run_system = st.sidebar.checkbox("Run System", value=True)

        while cap.isOpened() and run_system:
            ret, frame = cap.read()
            if not ret:
                st.info("Video feed end ho gayi ya source disconnect ho gaya.")
                break

            frame = cv2.resize(frame, (850, 480))
            results = model(frame, verbose=False)

            current_frame_speeds = []
            total_objects_in_frame = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
    
                    if cls in [2, 3, 5, 7]:
                        total_objects_in_frame += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        width = x2 - x1

                        distance = 5000 / max(width, 1)
                        speed = np.random.randint(30, 110) 
                        current_frame_speeds.append(speed)
                        speed_history.append(speed)

                        risk_score = get_fuzzy_risk(distance, speed)
                        risk_history.append(risk_score)

                        if risk_score >= 65:
                            label, color = "HIGH", (0, 0, 255)
                            risk_counts["HIGH"] += 1
                        elif risk_score >= 40:
                            label, color = "MEDIUM", (0, 165, 255)
                            risk_counts["MEDIUM"] += 1
                        else:
                            label, color = "LOW", (0, 255, 0)
                            risk_counts["LOW"] += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ({risk_score:.0f}%)", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if len(speed_history) > 100:
                speed_history.pop(0)
                risk_history.pop(0)

            avg_spd = np.mean(current_frame_speeds) if current_frame_speeds else 0
            max_rsk = np.max(risk_history) if risk_history else 0

            metric_total.metric(label="🚗 Vehicles in Frame", value=total_objects_in_frame)
            metric_avg_speed.metric(label="⚡ Avg Speed (km/h)", value=f"{avg_spd:.1f}")
            metric_max_risk.metric(label="⚠️ Max Risk Score", value=f"{max_rsk:.1f}%")

            if max_rsk > 70:
                metric_status.markdown("### 🚨 **CRITICAL STATUS**")
            else:
                metric_status.markdown("### ✅ **NORMAL STATUS**")


            fig = px.bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                labels={'x': 'Risk Category', 'y': 'Cumulative Alerts'},
                title="Advanced Risk Distribution",
                color=list(risk_counts.keys()),
                color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(t=40, b=10, l=10, r=10))
            chart_holder.plotly_chart(fig, use_container_width=True)


            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame_holder.image(rgb_frame, channels="RGB")

        cap.release()
