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
import base64
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""
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
    
    /* Glassmorphism Card styling */
    .glass-card {
        background: rgba(17, 25, 40, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 230, 118, 0.4);
        box-shadow: 0 12px 40px 0 rgba(0, 230, 118, 0.15);
    }
    .glass-card.contaminated:hover {
        border: 1px solid rgba(255, 65, 54, 0.4);
        box-shadow: 0 12px 40px 0 rgba(255, 65, 54, 0.15);
    }
    
    .download-link {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff !important;
        padding: 6px 12px;
        border-radius: 6px;
        text-decoration: none;
        font-size: 13px;
        font-weight: 600;
        transition: all 0.2s;
        margin-top: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        width: 100%;
        text-align: center;
    }
    .download-link:hover {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000000 !important;
        border-color: transparent;
        box-shadow: 0px 4px 15px rgba(0, 201, 255, 0.3);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)
# Initialize session state for analytics
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None
if 'selected_sample_name' not in st.session_state:
    st.session_state.selected_sample_name = None
if 'trigger_diagnosis' not in st.session_state:
    st.session_state.trigger_diagnosis = False
if 'active_result' not in st.session_state:
    st.session_state.active_result = None
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
# --- FEATURE 4: About This Project sidebar panel ---
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This Project", expanded=False):
    st.markdown("""
    <div style="font-size: 13px; line-height: 1.4;">
        <p><strong>Microplastic Matrix Pro</strong> is an optical analysis dashboard utilizing Deep Learning to monitor synthetic microparticles in water systems.</p>
        <ul>
            <li><strong>Engine:</strong> CNN-based Classifier</li>
            <li><strong>Frameworks:</strong> TensorFlow / Streamlit</li>
            <li><strong>Explainability:</strong> Grad-CAM heatmap</li>
            <li><strong>Mission:</strong> Real-time environmental monitoring</li>
        </ul>
        <hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);"/>
        <p style="font-size:11px; color:#888;">Optimized for deployment on Hugging Face Spaces.</p>
    </div>
    """, unsafe_allow_html=True)
if model is None:
    st.error("Model Error: Ensure 'model4.keras' is configured correctly.")
# --- MODULES --- #
st.title(mode)
if mode == "🧪 Single Sample Lab":
    st.write("Precision deep-learning analysis with Explainable AI.")
    
    # Initialize quick test state
    uploaded_file = st.file_uploader("Upload Sample", type=["jpg", "png"])
    
    # If the user uploads a new file, clear any quick test selection
    if uploaded_file:
        st.session_state.selected_sample = None
        st.session_state.selected_sample_name = None
        
    active_image = None
    active_filename = None
    active_source = "Upload"
    
    if uploaded_file:
        active_image = Image.open(uploaded_file)
        active_filename = uploaded_file.name
        active_source = "Upload"
    elif st.session_state.selected_sample:
        active_image = Image.open(st.session_state.selected_sample)
        active_filename = st.session_state.selected_sample_name
        active_source = "Quick Test"
        
    if active_image and model:
        # Determine if we should run prediction instantly
        run_diagnosis = False
        if st.session_state.trigger_diagnosis:
            run_diagnosis = True
            st.session_state.trigger_diagnosis = False # Reset trigger
        elif active_source == "Quick Test":
            # Auto-run for quick test if not already run
            if not st.session_state.active_result or st.session_state.active_result["filename"] != active_filename:
                run_diagnosis = True
                
        if run_diagnosis:
            with st.spinner("Analyzing Morphology..."):
                conf, p_img = run_diagnostics(active_image, active_filename, active_source)
                
                # Get last conv layer for Grad-CAM
                last_conv = get_last_conv_layer_name(model)
                heatmap_img = None
                if last_conv:
                    heatmap = make_gradcam_heatmap(p_img, model, last_conv)
                    if heatmap is not None:
                        heatmap_img = overlay_heatmap(heatmap, active_image)
                        
                st.session_state.active_result = {
                    "filename": active_filename,
                    "confidence": conf,
                    "heatmap": heatmap_img
                }
            st.rerun()
            
        # Display the result side-by-side if it exists
        if st.session_state.active_result and st.session_state.active_result["filename"] == active_filename:
            res = st.session_state.active_result
            col1, col2 = st.columns(2)
            with col1:
                st.image(active_image, use_column_width=True, caption=f"Active Sample: {res['filename']}")
                if active_source == "Quick Test":
                    if st.button("🧹 Clear Quick Test"):
                        st.session_state.selected_sample = None
                        st.session_state.selected_sample_name = None
                        st.session_state.active_result = None
                        st.rerun()
            with col2:
                st.write("### AI Analysis Results")
                st.progress(res["confidence"])
                st.write(f"**Contamination Probability: {res['confidence']*100:.1f}%**")
                if res["confidence"] > 0.5:
                    st.error("🚨 Microplastics Detected")
                else:
                    st.success("✅ Clean")
                
                if res["heatmap"] is not None:
                    st.write("### Explainable AI Focus Map")
                    st.image(res["heatmap"], use_column_width=True, caption="Grad-CAM Activation Map (highlights features used by the model)")
        else:
            # Show original image and a button to run the diagnosis
            col1, col2 = st.columns(2)
            with col1:
                st.image(active_image, use_column_width=True, caption=f"Original Sample: {active_filename}")
                if active_source == "Quick Test":
                    if st.button("🧹 Clear Quick Test"):
                        st.session_state.selected_sample = None
                        st.session_state.selected_sample_name = None
                        st.session_state.active_result = None
                        st.rerun()
            with col2:
                st.write("🔬 Sample loaded. Click the button below to analyze.")
                if st.button("Run Deep Diagnosis", type="primary", key="run_diag_manual"):
                    st.session_state.trigger_diagnosis = True
                    st.rerun()
    # --- FEATURE 1: SAMPLE TEST IMAGES ---
    st.write("")
    st.write("")
    st.markdown("---")
    st.subheader("🧪 Sample Test Images")
    st.write("Select one of the microscopic samples below to test the CNN model immediately:")
    
    samples_data = [
        {
            "path": "samples/clean1.jpg",
            "name": "clean1.jpg",
            "label": "Clean Water Sample",
            "desc": "Microscopic scan of purified tap water containing zero plastic particulates.",
            "class": "clean"
        },
        {
            "path": "samples/clean2.jpg",
            "name": "clean2.jpg",
            "label": "Clean Water Sample",
            "desc": "Filtered spring water control sample with standard mineral signature.",
            "class": "clean"
        },
        {
            "path": "samples/contaminated1.jpg",
            "name": "contaminated1.jpg",
            "label": "Microplastic Detected Sample",
            "desc": "Marine water containing high density of synthetic polyethylene fragments.",
            "class": "contaminated"
        },
        {
            "path": "samples/contaminated2.jpg",
            "name": "contaminated2.jpg",
            "label": "Microplastic Detected Sample",
            "desc": "Tap water contaminated with microscopic polystyrene microbeads.",
            "class": "contaminated"
        }
    ]
    
    cols = st.columns(4)
    for i, sample in enumerate(samples_data):
        with cols[i]:
            b64 = get_image_base64(sample["path"])
            border_class = "contaminated" if sample["class"] == "contaminated" else ""
            label_color = "#EF553B" if sample["class"] == "contaminated" else "#00e676"
            
            st.markdown(f"""
            <div class="glass-card {border_class}">
                <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:120px; border-radius:8px; object-fit:cover; border: 1px solid rgba(255,255,255,0.05);"/>
                <h5 style="color:{label_color}; margin: 10px 0 5px 0; font-size:13px; font-weight:700;">{sample["label"]}</h5>
                <p style="font-size:11px; color:#cccccc; height: 50px; overflow: hidden; margin-bottom: 10px; line-height: 1.3;">{sample["desc"]}</p>
                <a href="data:image/jpeg;base64,{b64}" download="{sample["name"]}" class="download-link">📥 Download</a>
            </div>
            """, unsafe_allow_html=True)
            
            # Test button under the card
            # if st.button(f"⚡ Test Sample {i+1}", key=f"quick_test_btn_{i}", use_container_width=True):
            #     st.session_state.selected_sample = sample["path"]
            #     st.session_state.selected_sample_name = sample["name"]
            #     st.session_state.trigger_diagnosis = True
            #     st.rerun()
    # --- FEATURE 3: QUICK TEST BUTTONS (below the grid) ---
    # st.write("")
    # col_btn1, col_btn2 = st.columns(2)
    # with col_btn1:
    #     if st.button("🧪 Test Clean Sample", use_container_width=True):
    #         st.session_state.selected_sample = "samples/clean1.jpg"
    #         st.session_state.selected_sample_name = "clean1.jpg"
    #         st.session_state.trigger_diagnosis = True
    #         st.rerun()
    # with col_btn2:
    #     if st.button("🚨 Test Contaminated Sample", use_container_width=True):
    #         st.session_state.selected_sample = "samples/contaminated1.jpg"
    #         st.session_state.selected_sample_name = "contaminated1.jpg"
    #         st.session_state.trigger_diagnosis = True
    #         st.rerun()
    # --- FEATURE 2: HOW TO USE SECTION ---
    st.write("")
    st.write("")
    st.markdown("---")
    st.subheader("📘 How To Use")
    
    with st.expander("📖 Step-by-Step Analysis Protocol Guide", expanded=False):
        st.markdown("""
        <div style="padding: 10px; font-family: 'Inter', sans-serif;">
            <div style="margin-bottom: 20px; display: flex; align-items: flex-start; gap: 15px;">
                <div style="background: rgba(0, 201, 255, 0.1); border: 1px solid #00C9FF; border-radius: 50%; min-width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #00C9FF;">1</div>
                <div>
                    <h5 style="color: #00C9FF; margin: 0; font-size:15px; font-weight:600;">📤 Upload an Image / Choose Sample</h5>
                    <p style="font-size: 13px; margin: 4px 0 0 0; color: #aaaaaa; line-height:1.4;">Click <strong>Browse Files</strong> in the upload box or select one of the pre-loaded <strong>Sample Test Images</strong> below.</p>
                </div>
            </div>
            <div style="margin-bottom: 20px; display: flex; align-items: flex-start; gap: 15px;">
                <div style="background: rgba(0, 201, 255, 0.1); border: 1px solid #00C9FF; border-radius: 50%; min-width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #00C9FF;">2</div>
                <div>
                    <h5 style="color: #00C9FF; margin: 0; font-size:15px; font-weight:600;">🧠 AI Deep Learning Processing</h5>
                    <p style="font-size: 13px; margin: 4px 0 0 0; color: #aaaaaa; line-height:1.4;">Click <strong>Run Deep Diagnosis</strong> (automatically runs for sample images). The custom Convolutional Neural Network (CNN) model parses the optical morphology of the liquid sample.</p>
                </div>
            </div>
            <div style="margin-bottom: 20px; display: flex; align-items: flex-start; gap: 15px;">
                <div style="background: rgba(0, 201, 255, 0.1); border: 1px solid #00C9FF; border-radius: 50%; min-width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #00C9FF;">3</div>
                <div>
                    <h5 style="color: #00C9FF; margin: 0; font-size:15px; font-weight:600;">🎯 Prediction Result</h5>
                    <p style="font-size: 13px; margin: 4px 0 0 0; color: #aaaaaa; line-height:1.4;">The system automatically categorizes the sample state as either <span style="color:#00e676; font-weight:600;">Clean Water</span> or <span style="color:#EF553B; font-weight:600;">Microplastic Contaminated</span>.</p>
                </div>
            </div>
            <div style="margin-bottom: 20px; display: flex; align-items: flex-start; gap: 15px;">
                <div style="background: rgba(0, 201, 255, 0.1); border: 1px solid #00C9FF; border-radius: 50%; min-width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #00C9FF;">4</div>
                <div>
                    <h5 style="color: #00C9FF; margin: 0; font-size:15px; font-weight:600;">📈 Confidence Score</h5>
                    <p style="font-size: 13px; margin: 4px 0 0 0; color: #aaaaaa; line-height:1.4;">A real-time progress bar shows the mathematical probability of contamination detected by the deep learning layers.</p>
                </div>
            </div>
            <div style="display: flex; align-items: flex-start; gap: 15px;">
                <div style="background: rgba(0, 201, 255, 0.1); border: 1px solid #00C9FF; border-radius: 50%; min-width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #00C9FF;">5</div>
                <div>
                    <h5 style="color: #00C9FF; margin: 0; font-size:15px; font-weight:600;">🔬 Explainable AI (Grad-CAM Focus Map)</h5>
                    <p style="font-size: 13px; margin: 4px 0 0 0; color: #aaaaaa; line-height:1.4;">The system runs Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the specific pixels and regions that influenced the model's classification decision.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
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
# --- FEATURE 5: UNIVERSAL FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px 0; color: #888888; font-size: 13px; font-family: 'Inter', sans-serif;">
    <p style="margin: 0; font-weight: 500;">🔬 Developed for Environmental AI Research</p>
    <p style="margin: 5px 0 15px 0;">Powered by <strong>TensorFlow</strong> • <strong>Streamlit</strong> • <strong>Hugging Face</strong></p>
    <div style="display: flex; justify-content: center; gap: 15px; font-size: 12px;">
        <a href="https://github.com" target="_blank" style="color: #00e676; text-decoration: none; font-weight: 600; transition: color 0.2s;">💻 GitHub Repository</a>
        <span style="color: #444;">|</span>
        <a href="https://huggingface.co" target="_blank" style="color: #00C9FF; text-decoration: none; font-weight: 600; transition: color 0.2s;">🤗 Hugging Face Space</a>
    </div>
</div>
""", unsafe_allow_html=True)