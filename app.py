import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

# ==========================================
# 1. CORE VISION-LANGUAGE ALIGNMENT NETWORK
# ==========================================
class SatQueryVLM(nn.Module):
    """
    A conceptual Multi-task Vision-Language Model framework for Remote Sensing.
    Processes spatial pixels and text tokens to handle multiple queries dynamically.
    """
    def __init__(self):
        super(SatQueryVLM, self).__init__()
        # Visual feature extractor backbone
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Spatial Decoder for dynamic multi-class classification or masks
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, img_tensor):
        features = self.image_encoder(img_tensor)
        mask = self.segmentation_head(features)
        return mask

# ==========================================
# 2. INTELLIGENT TEXT COGNITION SIMULATION
# ==========================================
def generate_satquery_insights(query_text):
    """
    Simulates the VLM text decoder engine generating targeted environmental
    assessments according to user query intents.
    """
    q = query_text.lower()
    if "urban" in q or "building" in q or "infrastructure" in q:
        return ("### 🏙️ SatQuery AI Structural Assessment Report\n"
                "- **Detected Artifacts:** 14 high-density residential concrete assets found.\n"
                "- **Geospatial Layout:** Structural expansion identified along the eastern corridor.\n"
                "- **Anomaly Flags:** No structural breaches or illegal encroachment caught.")
    elif "farm" in q or "crop" in q or "agriculture" in q:
        return ("### 🌾 SatQuery AI Agricultural Canopy Report\n"
                "- **Estimated NDVI Density:** Healthy normalized vegetative index detected (0.68 - 0.75).\n"
                "- **Soil Diagnostics:** Moderate moisture patterns observed via infrared signatures.\n"
                "- **Harvest Status:** Harvesting activity noted across northern parcels.")
    else:
        return ("### 🌍 SatQuery AI General Regional Mapping\n"
                "- **Primary Classification:** Mixed terrain configuration containing vegetation and soil indices.\n"
                "- **Recommendation:** Refine text strings to specify infrastructure, forest, or water classifications.")

# ==========================================
# 3. INTERACTIVE DASHBOARD TERMINAL
# ==========================================
st.set_page_config(layout="wide", page_title="ISRO SatQuery AI Engine")
st.title("🛰️ SatQuery AI: Interactive Vision-Language Assistant")
st.caption("ISRO Hackathon Track Prototype — Problem Statement ID: SIH26167")

# Dashboard Setup Sidebar
st.sidebar.header("📡 Satellite Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Remote Sensing Map (TIF / PNG / JPG)", type=["png", "jpg", "tiff"])
user_prompt = st.text_input("💬 Ask SatQuery AI about this region...", placeholder="e.g., Identify urban infrastructure development areas")

# Handle mock image logic if user hasn't provided a file
if uploaded_file:
    raw_img = Image.open(uploaded_file).convert('RGB')
    base_img = np.array(raw_img)
else:
    # Build a simulated multi-spectral landscape matrix
    base_img = np.zeros((400, 400, 3), dtype=np.uint8)
    base_img[:, :200] = [34, 139, 34]    # Forest green zone
    base_img[:, 200:] = [128, 128, 128]  # Urban grey zone
    cv2.putText(base_img, "FOREST REGION", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(base_img, "URBAN SECTOR", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

if st.button("⚡ Query Satellite Engine", use_container_width=True):
    if not user_prompt:
        st.warning("Please type a natural language prompt message to execute cross-modal analysis.")
    else:
        with st.spinner("Aligning vision tokens with language embeddings..."):
            
            # Run vision network prediction
            model = SatQueryVLM()
            model.eval()
            
            # Format image data array matrices into normalized tensors
            img_resized = cv2.resize(base_img, (400, 400))
            img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            with torch.no_grad():
                predicted_mask = model(img_tensor).squeeze().numpy()
            
            # Map dynamic target segment overlays based on prompt queries
            binary_mask = (predicted_mask > 0.4).astype(np.uint8) * 255
            if "urban" in user_prompt.lower() or "building" in user_prompt.lower():
                # Focus segmentation visualization onto the urban coordinates for simulation
                binary_mask[:, :200] = 0 
            elif "farm" in user_prompt.lower() or "crop" in user_prompt.lower() or "forest" in user_prompt.lower():
                binary_mask[:, 200:] = 0
                
            # Process analytical summary texts
            text_insights = generate_satquery_insights(user_prompt)
            
            # Render multi-column processing output window
            col1, col2, col3 = st.columns([1, 1, 1.2])
            
            with col1:
                st.image(base_img, caption="Ingested Satellite Stream", use_container_width=True)
            with col2:
                # Merge segmentation maps onto live images using colored highlight masks
                highlight_overlay = base_img.copy()
                highlight_overlay[binary_mask == 255] = [255, 0, 0] # Highlight target regions in red
                st.image(highlight_overlay, caption="Dynamic Localized Segments", use_container_width=True)
            with col3:
                st.info("🤖 VLM Generative Text Insight")
                st.markdown(text_insights)
