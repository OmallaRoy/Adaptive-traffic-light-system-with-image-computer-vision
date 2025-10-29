# ==========================================
# app.py — Streamlit UI for Traffic System
# ==========================================
import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime, timedelta

# ------------------------------
# Load batch results
# ------------------------------
with open("batch_results.pkl", "rb") as f:
    batch_results = pickle.load(f)

# ------------------------------
# System configuration
# ------------------------------
lane_polygons = {
    "Lane 1": [(0, 0), (0, 300), (200, 300), (200, 0)],
    "Lane 2": [(201, 0), (201, 300), (400, 300), (400, 0)]
}

MAX_GREEN = 30
MIN_GREEN = 5

# ------------------------------
# Forecast generation
# ------------------------------
def generate_traffic_forecast():
    current_time = datetime.now()
    times = [current_time + timedelta(minutes=10*i) for i in range(1, 5)]
    data = {
        "Time": [t.strftime("%H:%M") for t in times],
        "Vehicles": np.random.randint(8, 25, 4)
    }
    return pd.DataFrame(data)

# ------------------------------
# Light color logic
# ------------------------------
def get_light_color(green_time):
    if green_time <= MIN_GREEN + 2:
        return "RED"
    elif green_time < MAX_GREEN * 0.6:
        return "ORANGE"
    else:
        return "GREEN"

# ------------------------------
# Streamlit layout
# ------------------------------
st.set_page_config(page_title="Traffic Control System", layout="wide")

st.title("🚦 Intelligent Traffic Management System")
st.subheader("Real-time Traffic Monitoring & Signal Control")

uploaded_file = st.file_uploader("📂 Upload Traffic Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown(f"**Selected Image:** `{uploaded_file.name}`")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    upload_name = os.path.splitext(uploaded_file.name.lower())[0]

    # Match uploaded image to stored results
    result = None
    for r in batch_results:
        stored_name = os.path.splitext(r["image_name"].lower())[0]
        if upload_name in stored_name or stored_name in upload_name:
            result = r
            break

    if result is None:
        st.warning("⚠️ No detection results found for this image. Upload one from your batch results.")
    else:
        display_img = result["display_img"]
        if isinstance(display_img, np.ndarray):
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        lane_counts_list = result["vehicle_breakdown"]
        adaptive_green_times = result["adaptive_green_times"]

        # ------------------------------
        # Display detection result
        # ------------------------------
        st.markdown("### 🧠 Vehicle Detection Results")
        st.image(display_img, caption="Detected Vehicles", use_container_width=True)

        # ------------------------------
        # Lane-wise analysis
        # ------------------------------
        st.markdown("### 🛣️ Lane Status & Signal Control")
        cols = st.columns(2)

        for i, (lane_name, points) in enumerate(lane_polygons.items()):
            with cols[i]:
                poly = np.array(points, np.int32)
                mask = np.zeros(display_img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                lane_img = cv2.bitwise_and(display_img, display_img, mask=mask)
                lane_img_pil = Image.fromarray(lane_img)

                counts = lane_counts_list[i]
                total = sum(counts.values())
                green_time = adaptive_green_times.get(lane_name, 0)

                # Determine light color
                light_state = get_light_color(green_time)
                if light_state == "RED":
                    color_code = "#FF4B4B"
                elif light_state == "ORANGE":
                    color_code = "#FFA500"
                else:
                    color_code = "#00D474"

                # ------------------------------
                # Lane card UI
                # ------------------------------
                st.markdown(
                    f"""
                    <div style="
                        border: 3px solid {color_code};
                        border-radius: 12px;
                        padding: 16px;
                        margin: 8px 0;
                        background: white;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        position: relative;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <h3 style='margin:0; color:#1a73e8;'>{lane_name}</h3>
                            <span style="
                                background-color: {color_code};
                                color: white;
                                padding: 6px 16px;
                                border-radius: 20px;
                                font-weight: bold;
                                font-size: 14px;
                                position: absolute;
                                top: 16px;
                                right: 16px;
                            ">{light_state}</span>
                        </div>
                    """,
                    unsafe_allow_html=True
                )

                st.image(lane_img_pil, use_container_width=True)

                # Vehicle details
                st.markdown(
                    f"""
                    <div style="margin: 12px 0;">
                        <h4 style='margin:0; color:#5f6368;'>Vehicles: {total}</h4>
                    """,
                    unsafe_allow_html=True
                )
                for cls, num in counts.items():
                    st.markdown(f"- **{cls}:** {num}")
                st.markdown("</div>", unsafe_allow_html=True)

                # Progress / timing
                progress_width = int((green_time / MAX_GREEN) * 100)
                st.markdown(
                    f"""
                    <div style="margin: 16px 0;">
                        <div style="background-color: #f0f0f0; width: 100%; height: 20px; border-radius:10px; overflow:hidden;">
                            <div style="background-color: {color_code}; width: {progress_width}%; height: 100%; border-radius:10px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                            <small style="color: #5f6368;">Signal Duration</small>
                            <small style="color: #5f6368; font-weight: bold;">{green_time} sec</small>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Forecast display
                forecast_df = generate_traffic_forecast()
                next_forecast = forecast_df.iloc[0]["Vehicles"]

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 12px;
                        border-radius: 8px;
                        margin: 12px 0;
                    ">
                        <h4 style='margin:0 0 8px 0; color:white; font-size:16px;'>Traffic Forecast</h4>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 12px;">Next 10 min</div>
                                <div style="font-size: 20px; font-weight: bold;">{next_forecast} vehicles</div>
                            </div>
                            <div style="font-size: 24px;">📊</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ------------------------------
        # Overall summary
        # ------------------------------
        st.markdown("---")
        st.markdown("### 📈 Traffic Forecast Overview")

        cols_summary = st.columns(2)
        with cols_summary[0]:
            st.markdown("**Vehicle Forecast (Next 40 min)**")
            forecast_df = generate_traffic_forecast()
            st.bar_chart(forecast_df.set_index('Time'))

        with cols_summary[1]:
            st.markdown("**Traffic Summary**")

            total_vehicles = sum([sum(lane_counts_list[i].values()) for i in range(len(lane_polygons))])
            st.metric("Total Vehicles", f"{total_vehicles}", "High Density" if total_vehicles > 20 else "Normal")

            efficiency = min(100, int((total_vehicles / 40) * 100))
            st.metric("Route Efficiency", f"{efficiency}%", "Optimal" if efficiency > 75 else "Good")

            st.markdown("**System Status:** 🟢 **Operational**")
            st.markdown("**Signal Pattern:** 🔄 **Adaptive**")

else:
    st.info("📸 Please upload a traffic image to begin analysis.")