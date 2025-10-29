# # app_streamlit.py
# """
# Streamlit frontend that uploads an image to a FastAPI prediction endpoint
# and displays a polished prediction UI.

# Usage:
#     streamlit run app_streamlit.py

# Config:
#     - Change PREDICT_URL to the address where your FastAPI is running, e.g.:
#       "http://localhost:8000/predict"
# """

# import io
# import json
# import requests
# import streamlit as st
# import pandas as pd
# from PIL import Image
# from datetime import datetime

# # -----------------------
# # Config
# # -----------------------
# PREDICT_URL = "http://localhost:8000/predict"   # <-- change if needed
# TOP_K = 3  # how many top predictions to highlight

# # -----------------------
# # Helpers
# # -----------------------
# def post_image_to_api(image_bytes: bytes, filename: str):
#     """
#     POST the image bytes to the FastAPI /predict endpoint.
#     Returns (success_bool, result_or_error_message).
#     """
#     files = {"file": (filename, image_bytes, "image/jpeg")}
#     try:
#         resp = requests.post(PREDICT_URL, files=files, timeout=15)
#     except requests.RequestException as e:
#         return False, f"Request failed: {e}"

#     if resp.status_code != 200:
#         # try to extract JSON error
#         try:
#             err = resp.json()
#         except Exception:
#             err = resp.text
#         return False, f"Server returned {resp.status_code}: {err}"

#     try:
#         return True, resp.json()
#     except Exception as e:
#         return False, f"Invalid JSON response: {e} - raw: {resp.text}"


# def pretty_probs_dict(probs: dict):
#     """Return a sorted DataFrame of probabilities descending."""
#     df = pd.DataFrame(list(probs.items()), columns=["class", "probability"])
#     df["probability"] = df["probability"].astype(float)
#     df = df.sort_values("probability", ascending=False).reset_index(drop=True)
#     return df


# def add_to_history(img_bytes: bytes, filename: str, response: dict):
#     """Store recent predictions in session state for quick recall."""
#     entry = {
#         "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
#         "filename": filename,
#         "predicted_class": response.get("predicted_class"),
#         "confidence": float(response.get("confidence", 0)),
#         "raw": response
#     }
#     if "history" not in st.session_state:
#         st.session_state.history = []
#     # keep latest first, limit history length
#     st.session_state.history.insert(0, entry)
#     st.session_state.history = st.session_state.history[:10]


# # -----------------------
# # Streamlit UI
# # -----------------------
# st.set_page_config(page_title="X-ray Classifier", page_icon="🩺", layout="centered")
# st.title("🩺 Chest X-ray Classifier — Demo")
# st.markdown(
#     """
#     Upload a chest X-ray image and get a predicted class (COVID / Normal / Viral Pneumonia).
#     This app sends the image to a FastAPI prediction endpoint and shows the result.
#     """
# )

# col1, col2 = st.columns([1, 1])

# with col1:
#     uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

#     st.write("")  # spacing
#     st.markdown("**Options**")
#     st.write(f"Prediction endpoint: `{PREDICT_URL}`")
#     st.write("Top-K shown:", TOP_K)

# with col2:
#     st.markdown("**Recent predictions**")
#     if "history" in st.session_state and st.session_state.history:
#         for h in st.session_state.history:
#             st.write(f"- **{h['predicted_class']}** ({h['confidence']:.2%}) — {h['filename']} — {h['time']}")
#     else:
#         st.info("No predictions yet. Upload an image to see results here.")

# st.divider()

# if uploaded_file is not None:
#     # Read bytes and prepare preview
#     file_bytes = uploaded_file.read()
#     try:
#         img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#     except Exception as e:
#         st.error(f"Could not read image: {e}")
#         st.stop()

#     # Show preview and basic info
#     st.subheader("Preview")
#     st.image(img, use_column_width=True, caption=uploaded_file.name)
#     st.write(f"Image size: {img.size[0]} x {img.size[1]}   •   Format: {img.format or 'unknown'}")
#     st.write("")

#     if st.button("Predict", type="primary"):
#         with st.spinner("Sending image to model and waiting for prediction..."):
#             ok, result = post_image_to_api(file_bytes, uploaded_file.name)
#         if not ok:
#             st.error(result)
#         else:
#             # success: pretty display
#             predicted = result.get("predicted_class", "N/A")
#             confidence = float(result.get("confidence", 0.0))
#             probabilities = result.get("probabilities", {})

#             st.success(f"Predicted: **{predicted}** — confidence **{confidence:.2%}**")
#             st.write("---")

#             # Top-k bar chart
#             df_probs = pretty_probs_dict(probabilities)
#             topk = df_probs.head(TOP_K)
#             st.subheader("Top predictions")
#             st.table(topk.style.format({"probability": "{:.2%}"}))

#             # Horizontal bar chart
#             st.bar_chart(topk.set_index("class")["probability"])

#             # Raw JSON (collapsible)
#             st.subheader("Raw response")
#             st.code(json.dumps(result, indent=2), language="json")

#             # Download JSON button
#             json_bytes = json.dumps(result, indent=2).encode("utf-8")
#             st.download_button("Download result (JSON)", data=json_bytes, file_name=f"prediction_{uploaded_file.name}.json", mime="application/json")

#             # Add to history
#             add_to_history(file_bytes, uploaded_file.name, result)

# else:
#     st.info("Upload an image to get a prediction.")

# st.markdown("---")
# st.caption("Built with FastAPI (backend) + Streamlit (frontend). Adjust `PREDICT_URL` at top if your API is hosted elsewhere.")



# # app_streamlit_beautiful.py
# """
# Beautiful Streamlit frontend for X-ray classifier.
# Run: streamlit run app_streamlit_beautiful.py
# """

# import io
# import json
# import requests
# from datetime import datetime
# from pathlib import Path

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # ----------------------- #
# # Config
# # ----------------------- #
# PREDICT_URL_DEFAULT = "http://localhost:8000/predict"
# TOP_K_DEFAULT = 3
# DEMO_IMAGES_DIR = "demo_images"

# # ----------------------- #
# # Helper Functions
# # ----------------------- #
# def post_image_to_api(image_bytes, filename, predict_url, timeout=15):
#     """Send image to FastAPI backend for prediction."""
#     try:
#         files = {"file": (filename, image_bytes, "image/jpeg")}
#         resp = requests.post(predict_url, files=files, timeout=timeout)
#         resp.raise_for_status()
#         return True, resp.json()
#     except requests.exceptions.RequestException as e:
#         return False, f"Request failed: {e}"
#     except Exception as e:
#         return False, f"Invalid response: {e}"

# def pretty_probs_df(probs):
#     """Convert dict of probabilities to sorted DataFrame."""
#     df = pd.DataFrame(list(probs.items()), columns=["class", "probability"])
#     df["probability"] = df["probability"].astype(float)
#     return df.sort_values("probability", ascending=False).reset_index(drop=True)

# def add_to_history(image_bytes, filename, response):
#     """Store recent predictions in session state."""
#     entry = {
#         "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
#         "filename": filename,
#         "predicted_class": response.get("predicted_class", "N/A"),
#         "confidence": float(response.get("confidence", 0)),
#         "probabilities": response.get("probabilities", {}),
#         "image_bytes": image_bytes,
#     }
#     st.session_state.setdefault("history", [])
#     st.session_state.history.insert(0, entry)
#     st.session_state.history = st.session_state.history[:8]

# def load_demo_images(dirpath):
#     p = Path(dirpath)
#     return [str(f) for f in sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))] if p.exists() else []

# # ----------------------- #
# # Page Setup
# # ----------------------- #
# st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🩺", layout="wide")

# st.markdown("""
# <style>
#     .block-container {padding-top: 1.5rem;}
#     .card {background-color: #fff; padding: 1.2rem; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.05);}
#     .small {font-size: 13px; color: #6b7280;}
#     .pred-label {font-size: 22px; font-weight: 600;}
# </style>
# """, unsafe_allow_html=True)

# # ----------------------- #
# # Sidebar
# # ----------------------- #
# with st.sidebar:
#     st.header("⚙️ Settings")
#     predict_url = st.text_input("Prediction Endpoint", PREDICT_URL_DEFAULT)
#     top_k = st.slider("Show Top-K Predictions", 1, 6, TOP_K_DEFAULT)

#     st.markdown("### 🧩 Demo Images")
#     demo_imgs = load_demo_images(DEMO_IMAGES_DIR)
#     demo_choice = st.selectbox("Select a Demo Image", ["-- None --"] + demo_imgs) if demo_imgs else None
#     st.markdown("---")
#     st.caption("Made for demo — adjust endpoint and upload new images.")

# # ----------------------- #
# # Header
# # ----------------------- #
# st.title("🩺 Chest X-ray Classifier")
# st.write("Upload a chest X-ray to classify (e.g., **COVID**, **Normal**, or **Viral Pneumonia**).")

# st.divider()

# # ----------------------- #
# # Image Upload & Prediction
# # ----------------------- #
# col1, col2 = st.columns([1.3, 1])

# with col1:
#     uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

#     # If demo image chosen
#     if demo_choice and demo_choice != "-- None --" and not uploaded_file:
#         with open(demo_choice, "rb") as f:
#             uploaded_file = io.BytesIO(f.read())
#             uploaded_file.name = Path(demo_choice).name

#     if uploaded_file:
#         file_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file.getvalue()
#         filename = getattr(uploaded_file, "name", "uploaded.jpg")

#         # Preview Image
#         image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#         st.image(image, caption=f"🖼️ {filename}", use_container_width=True)
#         st.caption(f"**Resolution:** {image.size[0]}x{image.size[1]}  |  **Mode:** {image.mode}")

#         if st.button("🚀 Predict", type="primary", use_container_width=True):
#             with st.spinner("Sending to model..."):
#                 ok, result = post_image_to_api(file_bytes, filename, predict_url)
#             if ok:
#                 st.session_state["last_result"] = result
#                 st.session_state["last_image"] = file_bytes
#                 add_to_history(file_bytes, filename, result)
#                 st.success("Prediction received ✅")
#             else:
#                 st.error(result)

# with col2:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("📊 Prediction Results")

#     if "last_result" in st.session_state:
#         result = st.session_state["last_result"]
#         pred = result.get("predicted_class", "—")
#         conf = float(result.get("confidence", 0))
#         probs = result.get("probabilities", {})

#         st.markdown(f"<div class='pred-label'>{pred}</div>", unsafe_allow_html=True)
#         st.progress(conf)
#         st.metric("Confidence", f"{conf:.2%}")

#         # Top Predictions
#         df_probs = pretty_probs_df(probs)
#         st.markdown("**Top Predictions:**")
#         for _, row in df_probs.head(top_k).iterrows():
#             st.write(f"- {row['class']}: **{row['probability']:.2%}**")

#         # Probability Chart
#         fig = px.bar(df_probs, x="probability", y="class", orientation="h", text_auto=".1%")
#         fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
#         st.plotly_chart(fig, use_container_width=True)

#         # Raw JSON
#         with st.expander("🧾 View Raw Response"):
#             st.json(result)

#         st.download_button(
#             "💾 Download JSON",
#             data=json.dumps(result, indent=2).encode("utf-8"),
#             file_name=f"prediction_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json",
#             mime="application/json"
#         )
#     else:
#         st.info("Upload an image and click **Predict** to see results.")
#     st.markdown("</div>", unsafe_allow_html=True)

# st.divider()

# # ----------------------- #
# # Recent Predictions
# # ----------------------- #
# st.subheader("🕒 Recent Predictions")
# if st.session_state.get("history"):
#     cols = st.columns(4)
#     for i, h in enumerate(st.session_state.history):
#         with cols[i % 4]:
#             img = Image.open(io.BytesIO(h["image_bytes"])).resize((120, 120))
#             st.image(img, caption=h["predicted_class"], width=120)
#             st.caption(f"Conf: {h['confidence']:.1%}")
# else:
#     st.info("No recent predictions yet — your latest predictions will appear here.")

# st.caption("💡 Tip: Add sample images inside a folder named `demo_images/` for quick testing.")








# import io
# import json
# import requests
# from datetime import datetime
# from pathlib import Path
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # ------------------ #
# # Config
# # ------------------ #
# API_URL_DEFAULT = "http://localhost:8000/predict"
# TOP_K_DEFAULT = 3
# DEMO_IMAGES_DIR = "demo_images"

# # ------------------ #
# # Helpers
# # ------------------ #
# def predict_image(image_bytes, filename, api_url):
#     try:
#         files = {"file": (filename, image_bytes, "image/jpeg")}
#         resp = requests.post(api_url, files=files, timeout=15)
#         resp.raise_for_status()
#         return True, resp.json()
#     except Exception as e:
#         return False, f"⚠️ Prediction failed: {e}"

# def to_dataframe(probs):
#     df = pd.DataFrame(probs.items(), columns=["Class", "Probability"])
#     return df.sort_values("Probability", ascending=False).reset_index(drop=True)

# # ------------------ #
# # Page Setup
# # ------------------ #
# st.set_page_config("🩺 X-ray Classifier", "🩺", layout="wide")

# # ------------------ #
# # Custom Dark Theme Styling
# # ------------------ #
# st.markdown("""
# <style>
#     body, .stApp {
#         background-color: #000000;
#         color: #FFFFFF;
#     }
#     .main {
#         padding-top: 1.5rem;
#         color: #FFFFFF;
#     }
#     .title {
#         text-align: center;
#         font-size: 2.4rem;
#         font-weight: 700;
#         color: #00BFFF;
#         margin-bottom: 0.3rem;
#     }
#     .subtitle {
#         text-align: center;
#         color: #A9A9A9;
#         margin-bottom: 2rem;
#         font-size: 1.1rem;
#     }
#     .card {
#         background-color: #111111;
#         border-radius: 14px;
#         box-shadow: 0 0 20px rgba(0,191,255,0.2);
#         padding: 1.5rem;
#         color: #FFFFFF;
#     }
#     .stButton button {
#         background-color: #00BFFF !important;
#         color: white !important;
#         border: none;
#         border-radius: 8px;
#         font-weight: 600;
#         padding: 0.5rem 1rem;
#     }
#     .stButton button:hover {
#         background-color: #1E90FF !important;
#     }
#     .stProgress > div > div > div > div {
#         background-color: #00BFFF;
#     }
#     .stMetric {
#         color: #FFFFFF !important;
#     }
#     hr {
#         border: 1px solid #333333;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ #
# # Header
# # ------------------ #
# st.markdown("<div class='title'>🩺 Chest X-ray Classifier</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload a chest X-ray image to identify its class (COVID / Normal / Viral Pneumonia)</div>", unsafe_allow_html=True)

# # ------------------ #
# # Sidebar
# # ------------------ #
# with st.sidebar:
#     st.header("⚙️ Settings")
#     api_url = st.text_input("Prediction Endpoint", API_URL_DEFAULT)
#     top_k = st.slider("Show Top-K Predictions", 1, 6, TOP_K_DEFAULT)
#     demo_images = [str(f) for f in Path(DEMO_IMAGES_DIR).glob("*.[jp][pn]g")] if Path(DEMO_IMAGES_DIR).exists() else []
#     demo_choice = st.selectbox("Choose Demo Image", ["-- None --"] + demo_images)
#     st.markdown("---")
#     st.caption("💡 Tip: Add demo images in a folder named `demo_images/`.")

# # ------------------ #
# # Main Content
# # ------------------ #
# col1, col2 = st.columns([1.2, 1])

# with col1:
#     with st.container():
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.subheader("📤 Upload Image")

#         uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

#         if demo_choice != "-- None --" and not uploaded_file:
#             with open(demo_choice, "rb") as f:
#                 uploaded_file = io.BytesIO(f.read())
#                 uploaded_file.name = Path(demo_choice).name

#         if uploaded_file:
#             image_bytes = uploaded_file.read()
#             img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             st.image(img, caption=f"{uploaded_file.name}", use_container_width=True)

#             if st.button("🚀 Predict", use_container_width=True):
#                 with st.spinner("Getting prediction..."):
#                     ok, result = predict_image(image_bytes, uploaded_file.name, api_url)
#                 if ok:
#                     st.session_state["result"] = result
#                     st.session_state["image_bytes"] = image_bytes
#                     st.success("✅ Prediction complete!")
#                 else:
#                     st.error(result)
#         else:
#             st.info("Please upload or select an image.")
#         st.markdown("</div>", unsafe_allow_html=True)

# with col2:
#     with st.container():
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.subheader("📊 Prediction Result")

#         if "result" in st.session_state:
#             result = st.session_state["result"]
#             pred_class = result.get("predicted_class", "—")
#             conf = float(result.get("confidence", 0))
#             probs = result.get("probabilities", {})

#             st.markdown(f"### 🧠 <span style='color:#00BFFF'>{pred_class}</span>", unsafe_allow_html=True)
#             st.progress(conf)
#             st.metric("Confidence", f"{conf:.1%}")

#             df = to_dataframe(probs)
#             st.dataframe(df.head(top_k), use_container_width=True, hide_index=True)

#             fig = px.bar(df.head(top_k), x="Probability", y="Class", orientation="h", text_auto=".1%")
#             fig.update_layout(
#                 height=250,
#                 margin=dict(l=0, r=0, t=10, b=10),
#                 paper_bgcolor="#111111",
#                 plot_bgcolor="#111111",
#                 font=dict(color="#FFFFFF")
#             )
#             st.plotly_chart(fig, use_container_width=True)

#             with st.expander("🔍 View Raw Response"):
#                 st.json(result)
#         else:
#             st.info("Prediction will appear here.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # ------------------ #
# # Footer
# # ------------------ #
# st.markdown("<hr>", unsafe_allow_html=True)
# st.caption("Built with ❤️ using Streamlit + FastAPI — Dark Mode Edition")







# import io
# import json
# import requests
# from datetime import datetime
# from pathlib import Path
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # ------------------ #
# # Config
# # ------------------ #
# API_URL_DEFAULT = "http://localhost:8000/predict"
# TOP_K_DEFAULT = 3
# DEMO_IMAGES_DIR = "demo_images"

# # ------------------ #
# # Helpers
# # ------------------ #
# def predict_image(image_bytes, filename, api_url):
#     try:
#         files = {"file": (filename, image_bytes, "image/jpeg")}
#         resp = requests.post(api_url, files=files, timeout=15)
#         resp.raise_for_status()
#         return True, resp.json()
#     except Exception as e:
#         return False, f"⚠️ Prediction failed: {e}"

# def to_dataframe(probs):
#     df = pd.DataFrame(probs.items(), columns=["Class", "Probability"])
#     return df.sort_values("Probability", ascending=False).reset_index(drop=True)

# # ------------------ #
# # Page Setup
# # ------------------ #
# st.set_page_config("🩺 X-ray Classifier", "🩺", layout="wide")

# # ------------------ #
# # Custom CSS
# # ------------------ #
# st.markdown("""
# <style>
#     body, .stApp {
#         background-color: #000000;
#         color: #ffffff;
#     }
#     .title {
#         text-align: center;
#         font-size: 2.5rem;
#         font-weight: 800;
#         color: #00b4d8;
#         margin-bottom: 0.3rem;
#     }
#     .subtitle {
#         text-align: center;
#         color: #cccccc;
#         margin-bottom: 2rem;
#     }
#     .card {
#         background-color: #111111;
#         border-radius: 16px;
#         box-shadow: 0 0 12px rgba(0, 180, 216, 0.2);
#         padding: 1.5rem;
#     }
#     .stButton>button {
#         background-color: #00b4d8 !important;
#         color: white !important;
#         font-weight: 600;
#         border-radius: 10px !important;
#         border: none;
#     }
#     .stButton>button:hover {
#         background-color: #0096c7 !important;
#     }
#     .stDataFrame {
#         background-color: #000000 !important;
#         color: white !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ #
# # Header
# # ------------------ #
# st.markdown("<div class='title'>🩺 Chest X-ray Classifier</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload a chest X-ray image to identify its class (COVID / Normal / Viral Pneumonia)</div>", unsafe_allow_html=True)

# # ------------------ #
# # Sidebar
# # ------------------ #
# with st.sidebar:
#     st.header("⚙️ Settings")
#     api_url = st.text_input("Prediction Endpoint", API_URL_DEFAULT)
#     top_k = st.slider("Show Top-K Predictions", 1, 6, TOP_K_DEFAULT)
#     demo_images = [str(f) for f in Path(DEMO_IMAGES_DIR).glob("*.[jp][pn]g")] if Path(DEMO_IMAGES_DIR).exists() else []
#     demo_choice = st.selectbox("Choose Demo Image", ["-- None --"] + demo_images)
#     st.markdown("---")
#     st.caption("💡 Tip: Add demo images in a folder named `demo_images/`.")

# # ------------------ #
# # Main Layout
# # ------------------ #
# col1, col2 = st.columns([1.2, 1])

# with col1:
#     with st.container():
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.subheader("📤 Upload Image")

#         uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

#         if demo_choice != "-- None --" and not uploaded_file:
#             with open(demo_choice, "rb") as f:
#                 uploaded_file = io.BytesIO(f.read())
#                 uploaded_file.name = Path(demo_choice).name

#         if uploaded_file:
#             image_bytes = uploaded_file.read()
#             img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             st.image(img, caption=f"{uploaded_file.name}", width="stretch")

#             if st.button("🚀 Predict", use_container_width=False):
#                 with st.spinner("Getting prediction..."):
#                     ok, result = predict_image(image_bytes, uploaded_file.name, api_url)
#                 if ok:
#                     st.session_state["result"] = result
#                     st.session_state["image_bytes"] = image_bytes
#                     st.success("✅ Prediction complete!")
#                 else:
#                     st.error(result)
#         else:
#             st.info("Please upload or select an image.")
#         st.markdown("</div>", unsafe_allow_html=True)

# with col2:
#     with st.container():
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.subheader("📊 Prediction Result")

#         if "result" in st.session_state:
#             result = st.session_state["result"]
#             pred_class = result.get("predicted_class", "—")
#             conf = float(result.get("confidence", 0))
#             probs = result.get("probabilities", {})

#             # Display top prediction clearly
#             st.markdown(f"<h3 style='color:#00b4d8;'>🧠 Top Prediction: {pred_class}</h3>", unsafe_allow_html=True)
#             st.metric("Confidence", f"{conf:.1%}")
#             st.progress(conf)

#             df = to_dataframe(probs)
#             st.dataframe(df.head(top_k), width="stretch", hide_index=True)

#             fig = px.bar(
#                 df.head(top_k),
#                 x="Probability",
#                 y="Class",
#                 orientation="h",
#                 text_auto=".1%",
#                 color="Probability",
#                 color_continuous_scale="teal"
#             )
#             fig.update_layout(
#                 height=280,
#                 margin=dict(l=0, r=0, t=10, b=10),
#                 plot_bgcolor="black",
#                 paper_bgcolor="black",
#                 font_color="white"
#             )
#             st.plotly_chart(fig, width="stretch")

#             with st.expander("🔍 View Raw Response"):
#                 st.json(result)
#         else:
#             st.info("Prediction will appear here.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # ------------------ #
# # Footer
# # ------------------ #
# st.markdown("<hr>", unsafe_allow_html=True)
# st.caption("Built with ❤️ using Streamlit + FastAPI")



# import io
# import requests
# from pathlib import Path
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # ------------------ #
# # Config
# # ------------------ #
# API_URL_DEFAULT = "http://localhost:8000/predict"
# TOP_K_DEFAULT = 3
# DEMO_IMAGES_DIR = "demo_images"

# # ------------------ #
# # Helpers
# # ------------------ #
# def predict_image(image_bytes, filename, api_url):
#     try:
#         files = {"file": (filename, image_bytes, "image/jpeg")}
#         resp = requests.post(api_url, files=files, timeout=15)
#         resp.raise_for_status()
#         return True, resp.json()
#     except Exception as e:
#         return False, f"⚠️ Prediction failed: {e}"

# def to_dataframe(probs):
#     df = pd.DataFrame(probs.items(), columns=["Class", "Probability"])
#     return df.sort_values("Probability", ascending=False).reset_index(drop=True)

# # ------------------ #
# # Page Setup
# # ------------------ #
# st.set_page_config("🩺 X-ray Classifier", "🩺", layout="wide")

# # ------------------ #
# # Custom CSS
# # ------------------ #
# st.markdown("""
# <style>
#     body, .stApp {
#         background-color: #000000;
#         color: #ffffff;
#     }
#     .title {
#         text-align: center;
#         font-size: 2.5rem;
#         font-weight: 800;
#         color: #00b4d8;
#         margin-bottom: 0.3rem;
#     }
#     .subtitle {
#         text-align: center;
#         color: #cccccc;
#         margin-bottom: 2rem;
#     }
#     .card {
#         background-color: #111111;
#         border-radius: 16px;
#         box-shadow: 0 0 18px rgba(0, 180, 216, 0.2);
#         padding: 1.5rem;
#         margin-bottom: 2rem;
#     }
#     .stButton>button {
#         background-color: #00b4d8 !important;
#         color: white !important;
#         font-weight: 600;
#         border-radius: 10px !important;
#         border: none;
#         width: 100%;
#     }
#     .stButton>button:hover {
#         background-color: #0096c7 !important;
#     }
#     .stDataFrame {
#         background-color: #000000 !important;
#         color: white !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ #
# # Header
# # ------------------ #
# st.markdown("<div class='title'>🩺 Chest X-ray Classifier</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload a chest X-ray image to identify its class (COVID / Normal / Viral Pneumonia)</div>", unsafe_allow_html=True)

# # ------------------ #
# # Sidebar
# # ------------------ #
# with st.sidebar:
#     st.header("⚙️ Settings")
#     api_url = st.text_input("Prediction Endpoint", API_URL_DEFAULT)
#     top_k = st.slider("Show Top-K Predictions", 1, 6, TOP_K_DEFAULT)
#     demo_images = [str(f) for f in Path(DEMO_IMAGES_DIR).glob("*.[jp][pn]g")] if Path(DEMO_IMAGES_DIR).exists() else []
#     demo_choice = st.selectbox("Choose Demo Image", ["-- None --"] + demo_images)
#     st.markdown("---")
#     st.caption("💡 Tip: Add demo images in a folder named `demo_images/`.")

# # ------------------ #
# # Upload Section
# # ------------------ #
# with st.container():
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("📤 Upload Image")

#     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

#     if demo_choice != "-- None --" and not uploaded_file:
#         with open(demo_choice, "rb") as f:
#             uploaded_file = io.BytesIO(f.read())
#             uploaded_file.name = Path(demo_choice).name

#     if uploaded_file:
#         image_bytes = uploaded_file.read()
#         img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         st.image(img, caption=f"{uploaded_file.name}", width="stretch")

#         if st.button("🚀 Predict"):
#             with st.spinner("Getting prediction..."):
#                 ok, result = predict_image(image_bytes, uploaded_file.name, api_url)
#             if ok:
#                 st.session_state["result"] = result
#                 st.session_state["image_bytes"] = image_bytes
#                 st.success("✅ Prediction complete!")
#             else:
#                 st.error(result)
#     else:
#         st.info("Please upload or select an image.")
#     st.markdown("</div>", unsafe_allow_html=True)

# # ------------------ #
# # Prediction Section (Below Image)
# # ------------------ #
# if "result" in st.session_state:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("📊 Prediction Result")

#     result = st.session_state["result"]
#     pred_class = result.get("predicted_class", "—")
#     conf = float(result.get("confidence", 0))
#     probs = result.get("probabilities", {})

#     # Highlight Top Prediction
#     st.markdown(f"<h3 style='color:#00b4d8;'>🧠 Top Prediction: {pred_class}</h3>", unsafe_allow_html=True)
#     st.metric("Confidence", f"{conf:.1%}")
#     st.progress(conf)

#     df = to_dataframe(probs)
#     st.dataframe(df.head(top_k), width="stretch", hide_index=True)

#     fig = px.bar(
#         df.head(top_k),
#         x="Probability",
#         y="Class",
#         orientation="h",
#         text_auto=".1%",
#         color="Probability",
#         color_continuous_scale="teal"
#     )
#     fig.update_layout(
#         height=280,
#         margin=dict(l=0, r=0, t=10, b=10),
#         plot_bgcolor="black",
#         paper_bgcolor="black",
#         font_color="white"
#     )
#     st.plotly_chart(fig, width="stretch")

#     with st.expander("🔍 View Raw Response"):
#         st.json(result)
#     st.markdown("</div>", unsafe_allow_html=True)

# # ------------------ #
# # Footer
# # ------------------ #
# st.markdown("<hr>", unsafe_allow_html=True)
# st.caption("Built with ❤️ using Streamlit + FastAPI")






# import io
# import requests
# from pathlib import Path
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # ------------------ #
# # Config
# # ------------------ #
# API_URL_DEFAULT = "http://localhost:8000/predict"

# # ------------------ #
# # Helpers
# # ------------------ #
# def predict_image(image_bytes, filename, api_url):
#     try:
#         files = {"file": (filename, image_bytes, "image/jpeg")}
#         resp = requests.post(api_url, files=files, timeout=15)
#         resp.raise_for_status()
#         return True, resp.json()
#     except Exception as e:
#         return False, f"⚠️ Prediction failed: {e}"

# def to_dataframe(probs):
#     df = pd.DataFrame(probs.items(), columns=["Class", "Probability"])
#     return df.sort_values("Probability", ascending=False).reset_index(drop=True)

# # ------------------ #
# # Page Setup
# # ------------------ #
# st.set_page_config("Chest X-ray Classifier", "🩺", layout="wide")

# # ------------------ #
# # Custom CSS
# # ------------------ #
# st.markdown("""
# <style>
#     body, .stApp {
#         background-color: #000000;
#         color: #ffffff;
#         font-family: 'Segoe UI', sans-serif;
#     }
#     .title {
#         text-align: center;
#         font-size: 2.5rem;
#         font-weight: 800;
#         color: #00b4d8;
#         margin-bottom: 0.3rem;
#     }
#     .subtitle {
#         text-align: center;
#         color: #aaaaaa;
#         margin-bottom: 2rem;
#         font-size: 1rem;
#     }
#     .card {
#         background-color: #111111;
#         border-radius: 16px;
#         box-shadow: 0 0 18px rgba(0, 180, 216, 0.15);
#         padding: 1.5rem;
#         margin-bottom: 2rem;
#     }
#     .stButton>button {
#         background-color: #00b4d8 !important;
#         color: white !important;
#         font-weight: 600;
#         border-radius: 10px !important;
#         border: none;
#         width: 100%;
#     }
#     .stButton>button:hover {
#         background-color: #0096c7 !important;
#     }
#     .stDataFrame {
#         background-color: #000000 !important;
#         color: white !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ #
# # Header
# # ------------------ #
# st.markdown("<div class='title'>🩺 Chest X-ray Classifier</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload a chest X-ray image to detect COVID-19, Viral Pneumonia, or Normal condition.</div>", unsafe_allow_html=True)

# # ------------------ #
# # Sidebar (minimal)
# # ------------------ #
# with st.sidebar:
#     st.header("⚙️ Settings")
#     api_url = st.text_input("Prediction Endpoint", API_URL_DEFAULT)
#     st.markdown("---")
#     st.caption("💡 Make sure FastAPI backend is running before prediction.")

# # ------------------ #
# # Upload Section
# # ------------------ #
# st.markdown("<div class='card'>", unsafe_allow_html=True)
# st.subheader("📤 Upload Image")

# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image_bytes = uploaded_file.read()
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
#     # Center the image in the middle with medium size
#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         st.image(img, caption=f"{uploaded_file.name}", width=400)  # medium width

#     if st.button("🚀 Predict"):
#         with st.spinner("Getting prediction..."):
#             ok, result = predict_image(image_bytes, uploaded_file.name, api_url)
#         if ok:
#             st.session_state["result"] = result
#             st.session_state["image_bytes"] = image_bytes
#             st.success("✅ Prediction complete!")
#         else:
#             st.error(result)
# else:
#     st.info("Please upload an image file to continue.")
# st.markdown("</div>", unsafe_allow_html=True)

# # ------------------ #
# # Prediction Section (below image)
# # ------------------ #
# if "result" in st.session_state:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("📊 Prediction Result")

#     result = st.session_state["result"]
#     pred_class = result.get("predicted_class", "—")
#     conf = float(result.get("confidence", 0))
#     probs = result.get("probabilities", {})

#     # Display top prediction prominently
#     st.markdown(f"<h3 style='color:#00b4d8;'>🧠 Predicted Class: <b>{pred_class}</b></h3>", unsafe_allow_html=True)
#     st.metric("Confidence", f"{conf:.1%}")
#     st.progress(conf)

#     # Show probabilities
#     df = to_dataframe(probs)
#     st.dataframe(df, use_container_width=True, hide_index=True)

#     fig = px.bar(
#         df,
#         x="Probability",
#         y="Class",
#         orientation="h",
#         text_auto=".1%",
#         color="Probability",
#         color_continuous_scale="teal"
#     )
#     fig.update_layout(
#         height=280,
#         margin=dict(l=0, r=0, t=10, b=10),
#         plot_bgcolor="black",
#         paper_bgcolor="black",
#         font_color="white"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     with st.expander("🔍 View Raw Response"):
#         st.json(result)
#     st.markdown("</div>", unsafe_allow_html=True)

# # ------------------ #
# # Footer
# # ------------------ #
# st.markdown("<hr>", unsafe_allow_html=True)
# st.caption("Built with using Streamlit + FastAPI")





import io
import requests
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import sys
import os

# ------------------ #
# Config
# ------------------ #
API_URL_DEFAULT = "http://localhost:8000/predict"

# ------------------ #
# Helpers
# ------------------ #
def predict_image(image_bytes, filename, api_url):
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        resp = requests.post(api_url, files=files, timeout=15)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as e:
        return False, f"⚠️ Prediction failed: {e}"

def to_dataframe(probs):
    df = pd.DataFrame(probs.items(), columns=["Class", "Probability"])
    return df.sort_values("Probability", ascending=False).reset_index(drop=True)

# ------------------ #
# Page Setup
# ------------------ #
st.set_page_config("Chest X-ray Classifier", "🩺", layout="wide")

# ------------------ #
# Custom CSS
# ------------------ #
st.markdown("""
<style>
    body, .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .title { text-align: center; font-size: 2.5rem; font-weight: 800; color: #00b4d8; margin-bottom: 0.3rem; }
    .subtitle { text-align: center; color: #aaaaaa; margin-bottom: 2rem; font-size: 1rem; }
    .card { background-color: #111111; border-radius: 16px; box-shadow: 0 0 18px rgba(0, 180, 216, 0.15); padding: 1.5rem; margin-bottom: 2rem; }
    .stButton>button { background-color: #00b4d8 !important; color: white !important; font-weight: 600; border-radius: 10px !important; border: none; width: 100%; }
    .stButton>button:hover { background-color: #0096c7 !important; }
    .stDataFrame { background-color: #000000 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ------------------ #
# Header
# ------------------ #
st.markdown("<div class='title'>🩺 Chest X-ray Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a chest X-ray image to detect COVID-19, Viral Pneumonia, or Normal condition.</div>", unsafe_allow_html=True)

# ------------------ #
# Sidebar
# ------------------ #
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("Prediction Endpoint", API_URL_DEFAULT)
    st.markdown("---")
    st.caption("💡 Make sure FastAPI backend is running before prediction.")

# ------------------ #
# Upload Section
# ------------------ #
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption=f"{uploaded_file.name}", width=400)

    if st.button("🚀 Predict"):
        with st.spinner("Getting prediction..."):
            ok, result = predict_image(image_bytes, uploaded_file.name, api_url)
        if ok:
            st.session_state["result"] = result
            st.session_state["image_bytes"] = image_bytes
            st.success("✅ Prediction complete!")
        else:
            st.error(result)
else:
    st.info("Please upload an image file to continue.")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------ #
# Prediction Section
# ------------------ #
if "result" in st.session_state:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    result = st.session_state["result"]
    pred_class = result.get("predicted_class", "—")
    conf = float(result.get("confidence", 0))
    probs = result.get("probabilities", {})

    st.markdown(f"<h3 style='color:#00b4d8;'>🧠 Predicted Class: <b>{pred_class}</b></h3>", unsafe_allow_html=True)
    st.metric("Confidence", f"{conf:.1%}")
    st.progress(conf)

    df = to_dataframe(probs)
    st.dataframe(df, width="stretch", hide_index=True)

    # ------------------ #
    # Plotly Bar Chart with stdout/stderr redirected to suppress warnings
    # ------------------ #
    class DummyFile:
        def write(self, x): pass
        def flush(self): pass

    dummy_stdout = sys.stdout
    dummy_stderr = sys.stderr
    sys.stdout = sys.stderr = DummyFile()  # redirect

    fig = px.bar(df, x="Probability", y="Class", orientation="h")
    fig.update_traces(
        text=df["Probability"].map(lambda x: f"{x:.1%}"),
        textposition="outside",
        marker=dict(color=df["Probability"], colorscale="teal")
    )
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white"
    )

    st.plotly_chart(fig, config={"displayModeBar": False, "responsive": True}, width="stretch")

    sys.stdout = dummy_stdout
    sys.stderr = dummy_stderr  # restore

    with st.expander("🔍 View Raw Response"):
        st.json(result)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ #
# Footer
# ------------------ #
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built using Streamlit + FastAPI")





