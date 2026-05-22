import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.graph_objects as go
import datetime
import random
import folium
from streamlit_folium import st_folium

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Microplastic Matrix Pro", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #00e676 !important; font-weight: 700; }
    /* Button Polish */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        border: none; border-radius: 8px; font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0px 4px 15px rgba(0, 201, 255, 0.4); }
    /* File uploader polish */
    .stFileUploader { background: rgba(0,255,0,0.02); border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for analytics
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_model():
    import json
    try:
        with open("model4.keras/config.json", "r") as f:
            config = json.load(f)
        model = tf.keras.models.model_from_json(json.dumps(config))
        model.load_weights("model4.keras/model.weights.h5")
        return model
    except Exception as e:
        return None

try:
    model = load_model()
except Exception as e:
    model = None

# --- Helpers for Grad-CAM --- #
def get_last_conv_layer_name(model):
    if not model: return None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    if not last_conv_layer_name: return None
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, 0]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_heatmap(heatmap, original_image, alpha=0.4):
    img = np.array(original_image)
    if img.shape[-1] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    if image.mode != "RGB": image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Mock coordinates near random water bodies
def get_random_location():
    return 34.00 + random.uniform(-2, 2), -118.00 + random.uniform(-2, 2)

def run_diagnostics(image_pil, filename, source="Upload"):
    processed_image = preprocess_image(image_pil)
    prediction = model.predict(processed_image)
    confidence = float(prediction[0][0])
    status = "Detected" if confidence > 0.5 else "Clean"
    lat, lon = get_random_location()
    
    st.session_state.history.append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Filename": filename,
        "Confidence": round(confidence, 4),
        "Status": status,
        "Lat": lat, "Lon": lon
    })
    return confidence, processed_image

# --- Sidebar UI Integration --- #
st.sidebar.title("🧬 Nexus Matrix")
mode = st.sidebar.radio("Select Analysis Module:", [
    "🧪 Single Sample Lab",
    "📸 Live Microscope Feed",
    "📁 Batch Processing Array",
    "⚖️ Comparative Analytics",
    "🗺️ Global Contamination Map"
])

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Analytics")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    contaminated = len(df[df['Status'] == 'Detected'])
    clean = len(df[df['Status'] == 'Clean'])
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("🚨 Detected", contaminated)
    col2.metric("✅ Clean", clean)
    
    fig = go.Figure(data=[go.Pie(labels=['Detected', 'Clean'], values=[contaminated, clean], hole=.5, marker_colors=['#EF553B', '#00CC96'])])
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=150, showlegend=False)
    st.sidebar.plotly_chart(fig, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Report CSV", data=csv, file_name='water_quality_report.csv', mime='text/csv')
else:
    st.sidebar.info("Awaiting data...")

if model is None:
    st.error("Model Error: Ensure 'model4.keras' is configured correctly.")

# --- MODULES --- #
st.title(mode)

if mode == "🧪 Single Sample Lab":
    st.write("Precision deep-learning analysis with Explainable AI.")
    uploaded_file = st.file_uploader("Upload Sample", type=["jpg", "png"])
    if uploaded_file and model:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, width=150, caption="Original Sample")
        with col2:
            if st.button("Run Deep Diagnosis", type="primary"):
                with st.spinner("Analyzing Morphology..."):
                    conf, p_img = run_diagnostics(img, uploaded_file.name)
                    st.progress(conf)
                    st.write(f"**Contamination Probability: {conf*100:.1f}%**")
                    if conf > 0.5: st.error("🚨 Microplastics Detected")
                    else: st.success("✅ Clean")
                    
                    last_conv = get_last_conv_layer_name(model)
                    if last_conv:
                        heatmap = make_gradcam_heatmap(p_img, model, last_conv)
                        if heatmap is not None:
                            overlay = overlay_heatmap(heatmap, img)
                            st.write("### AI Focus Map")
                            st.image(overlay, width=150, caption="Grad-CAM Activation")
                st.rerun()

elif mode == "📸 Live Microscope Feed":
    st.write("Real-time optical hardware integration via camera inputs.")
    cam_file = st.camera_input("Capture live sample")
    if cam_file and model:
        img = Image.open(cam_file)
        with st.spinner("Processing feed..."):
            conf, _ = run_diagnostics(img, "Live Feed", "Camera")
            if conf > 0.5: st.error(f"🚨 Microplastics Detected ({conf*100:.1f}%)")
            else: st.success(f"✅ Clean Sample ({conf*100:.1f}%)")

elif mode == "📁 Batch Processing Array":
    st.write("High-throughput analysis for multiple sample sets.")
    files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)
    if files and model:
        if st.button(f"Analyze {len(files)} Samples", type="primary"):
            progress_bar = st.progress(0)
            results = []
            for i, f in enumerate(files):
                img = Image.open(f)
                conf, _ = run_diagnostics(img, f.name, "Batch")
                results.append({"File": f.name, "Probability": f"{conf*100:.1f}%", "Status": "Detected" if conf > 0.5 else "Clean"})
                progress_bar.progress((i + 1) / len(files))
            st.success("Batch Processing Complete!")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

elif mode == "⚖️ Comparative Analytics":
    st.write("A/B test distinct liquid sources.")
    colA, colB = st.columns(2)
    with colA:
        file1 = st.file_uploader("Sample A (e.g. Tap Water)", type=["jpg", "png"], key="A")
    with colB:
        file2 = st.file_uploader("Sample B (e.g. River Water)", type=["jpg", "png"], key="B")
        
    if file1 and file2 and model and st.button("Compare Samples", type="primary"):
        with st.spinner("Analyzing comparative delta..."):
            conf1, _ = run_diagnostics(Image.open(file1), file1.name, "A/B Test")
            conf2, _ = run_diagnostics(Image.open(file2), file2.name, "A/B Test")
            
            cA, cB = st.columns(2)
            cA.metric("Sample A Probability", f"{conf1*100:.1f}%", delta="Contaminated" if conf1 > 0.5 else "Clean", delta_color="inverse")
            cB.metric("Sample B Probability", f"{conf2*100:.1f}%", delta="Contaminated" if conf2 > 0.5 else "Clean", delta_color="inverse")
            
            if conf1 < conf2: st.info("🏆 Sample A is safer.")
            elif conf2 < conf1: st.info("🏆 Sample B is safer.")

elif mode == "🗺️ Global Contamination Map":
    st.write("Geospatial plotting of Session history endpoints.")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        center_lat, center_lon = df['Lat'].mean(), df['Lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB dark_matter")
        
        for idx, row in df.iterrows():
            color = 'red' if row['Status'] == 'Detected' else 'green'
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=8,
                popup=f"{row['Filename']} - {row['Probability'] if 'Probability' in row else row['Confidence']}",
                color=color, fill=True, fill_color=color, fill_opacity=0.7
            ).add_to(m)
            
        st_folium(m, width=900, height=500)
    else:
        st.warning("No data available to map. Run analysis in other modules first.")

