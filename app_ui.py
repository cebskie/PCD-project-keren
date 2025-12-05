import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
import pickle
import warnings
import zipfile
import tempfile
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings('ignore')

# ============================================================================
# HIERARCHICAL CLASSIFIER CLASS
# ============================================================================

class HierarchicalClassifier:
    """
    Hierarchical classification:
    Stage 1: Shape features (initial screening)
    Stage 2: All features (refinement for uncertain cases)
    """

    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.stage1_model = None
        self.stage2_model = None
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.shape_cols = None
        self.all_cols = None
        self.is_fitted = False

    def fit(self, X, y):
        self.all_cols = X.columns.tolist()

        # Identify shape features
        self.shape_cols = [col for col in X.columns if any(
            keyword in col for keyword in ['area', 'perimeter', 'eccentricity',
                                          'extent', 'solidity', 'major', 'minor']
        )]

        X_shape = X[self.shape_cols]
        X_all = X

        # Scale features
        X_shape_scaled = self.scaler1.fit_transform(X_shape)
        X_all_scaled = self.scaler2.fit_transform(X_all)

        # Initialize models
        if self.model_type == 'svm':
            self.stage1_model = SVC(kernel='rbf', C=10, gamma='scale',
                                   probability=True, random_state=42,
                                   class_weight='balanced')
            self.stage2_model = SVC(kernel='rbf', C=10, gamma='scale',
                                   probability=True, random_state=42,
                                   class_weight='balanced')
        else:  # Naive Bayes
            self.stage1_model = GaussianNB()
            self.stage2_model = GaussianNB()

        # Train models
        self.stage1_model.fit(X_shape_scaled, y)
        self.stage2_model.fit(X_all_scaled, y)

        self.is_fitted = True

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X = X[self.all_cols]
        X_shape = X[self.shape_cols]

        X_shape_scaled = self.scaler1.transform(X_shape)
        X_all_scaled = self.scaler2.transform(X)

        # Stage 1: Quick screening with shape features
        stage1_pred = self.stage1_model.predict(X_shape_scaled)

        # Stage 2: Refinement with all features
        final_pred = stage1_pred.copy()
        for i in range(len(stage1_pred)):
            if stage1_pred[i] == 0:  # If predicted normal, double check
                stage2_pred = self.stage2_model.predict(X_all_scaled[i:i+1])
                final_pred[i] = stage2_pred[0]

        return final_pred

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X = X[self.all_cols]
        X_all_scaled = self.scaler2.transform(X)
        return self.stage2_model.predict_proba(X_all_scaled)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .header-title {
        font-size: 48px !important;
        color: #FCFCD !important;
        text-align: center !important;
        padding: 1rem !important;
        font-weight: bold !important;
        margin-bottom: 5 !important;
        margin-top: -3rem !important;
    }
    .result-card {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card h3 {
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }
    .result-card p {
        color: #2c3e50 !important;
        font-size: 20px !important;
        margin: 0.3rem 0 !important;
    }
    .normal-result {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .normal-result h3 {
        color: #155724 !important;
    }
    .tb-result {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .tb-result h3 {
        color: #721c24 !important;
    }
    .tb-result p strong {
        font-size: 16px !important;
        color: #721c24 !important;
    }
    .tb-result text{
        font-size: 12px !important;
        color: #faf7f7 !important;
        font-weight: bold !important;
    }
    .metric-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem 0 !important;
        color: #7f8c8d;
        border-top: 1px solid #ecf0f1;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models(base_path='.'):
    """Load trained models from disk."""
    svm_path = os.path.join(base_path, 'svm_model.pkl')
    nb_path = os.path.join(base_path, 'nb_model.pkl')
    
    # Check if model files exist
    if not os.path.exists(svm_path) or not os.path.exists(nb_path):
        st.error(f"❌ Model files not found in '{base_path}' folder!")
        st.info(f"Please ensure:\n- {svm_path}\n- {nb_path}")
        st.warning("Run Training_Model.ipynb to create the models first.")
        return None, None
    
    try:
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
        with open(nb_path, 'rb') as f:
            nb_model = pickle.load(f)
        
        return svm_model, nb_model
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

# ============================================================================
# PREPROCESSING & FEATURE EXTRACTION
# ============================================================================

def preprocess_image(img_array):
    """Preprocess image array."""
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    img_resized = cv2.resize(img_gray, (512, 512))
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return img_resized, enhanced

def segment_lungs(enhanced_img):
    """Segment lung regions."""
    _, binary = cv2.threshold(enhanced_img, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(enhanced_img)
    
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        num_lungs = min(2, len(contours))
        
        for i in range(num_lungs):
            if cv2.contourArea(contours[i]) > 1000:
                cv2.drawContours(mask, [contours[i]], -1, 255, -1)
    
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_final, iterations=2)
    
    return mask

def extract_shape_features(mask):
    """Extract shape features."""
    if mask is None or np.sum(mask) == 0:
        return {
            'area': 0, 'perimeter': 0, 'eccentricity': 0,
            'extent': 0, 'solidity': 0, 'major_axis': 0, 'minor_axis': 0
        }
    
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    
    if len(regions) == 0:
        return {
            'area': 0, 'perimeter': 0, 'eccentricity': 0,
            'extent': 0, 'solidity': 0, 'major_axis': 0, 'minor_axis': 0
        }
    
    total_area = sum([r.area for r in regions])
    total_perimeter = sum([r.perimeter for r in regions])
    avg_eccentricity = np.mean([r.eccentricity for r in regions])
    avg_extent = np.mean([r.extent for r in regions])
    avg_solidity = np.mean([r.solidity for r in regions])
    avg_major = np.mean([r.major_axis_length for r in regions])
    avg_minor = np.mean([r.minor_axis_length for r in regions])
    
    return {
        'area': total_area,
        'perimeter': total_perimeter,
        'eccentricity': avg_eccentricity,
        'extent': avg_extent,
        'solidity': avg_solidity,
        'major_axis': avg_major,
        'minor_axis': avg_minor
    }

def extract_first_order_stats(image, mask):
    """Extract statistical features."""
    if image is None or mask is None or np.sum(mask) == 0:
        return {k: 0 for k in ['mean', 'variance', 'std_dev', 'skewness',
                               'kurtosis', 'entropy', 'smoothness', 'uniformity']}
    
    masked_pixels = image[mask > 0]
    
    if len(masked_pixels) == 0:
        return {k: 0 for k in ['mean', 'variance', 'std_dev', 'skewness',
                               'kurtosis', 'entropy', 'smoothness', 'uniformity']}
    
    mean = np.mean(masked_pixels)
    variance = np.var(masked_pixels)
    std_dev = np.std(masked_pixels)
    skewness = stats.skew(masked_pixels)
    kurtosis = stats.kurtosis(masked_pixels)
    
    hist, _ = np.histogram(masked_pixels, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    smoothness = 1 - (1 / (1 + variance))
    uniformity = np.sum(hist ** 2)
    
    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'entropy': entropy,
        'smoothness': smoothness,
        'uniformity': uniformity
    }

def extract_glcm_features(image, mask):
    """Extract GLCM texture features."""
    if image is None or mask is None or np.sum(mask) == 0:
        return {k: 0 for k in ['contrast', 'dissimilarity', 'homogeneity',
                               'energy', 'correlation', 'ASM']}
    
    masked_img = image.copy()
    masked_img[mask == 0] = 0
    
    if np.all(masked_img == 0) or len(np.unique(masked_img)) < 2:
        return {k: 0 for k in ['contrast', 'dissimilarity', 'homogeneity',
                               'energy', 'correlation', 'ASM']}
    
    img_8bit = cv2.normalize(masked_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    try:
        glcm = graycomatrix(img_8bit, distances=distances, angles=angles,
                           levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        asm = graycoprops(glcm, 'ASM').mean()
        
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation,
            'ASM': asm
        }
    except:
        return {k: 0 for k in ['contrast', 'dissimilarity', 'homogeneity',
                               'energy', 'correlation', 'ASM']}

def extract_all_features(img_array):
    """Extract all features from image array."""
    original, enhanced = preprocess_image(img_array)
    mask = segment_lungs(enhanced)
    
    shape_features = extract_shape_features(mask)
    stats_features = extract_first_order_stats(enhanced, mask)
    glcm_features = extract_glcm_features(enhanced, mask)
    
    all_features = {**shape_features, **stats_features, **glcm_features}
    
    return all_features, original, enhanced, mask

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_image(image, model, filename="Image"):
    """Predict single image."""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Extract features
        features, original, enhanced, mask = extract_all_features(img_array)
        
        if features is None:
            return None
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        # Predict
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        result = {
            'filename': filename,
            'prediction': 'Normal' if prediction == 0 else 'Tuberculosis',
            'prediction_code': prediction,
            'confidence': probabilities[prediction] * 100,
            'prob_normal': probabilities[0] * 100,
            'prob_tb': probabilities[1] * 100,
            'original': original,
            'enhanced': enhanced,
            'mask': mask,
            'features': features
        }
        
        return result
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_images_from_folder(folder_path):
    """Extract all image files from folder."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def create_results_dataframe(results):
    """Create summary DataFrame from results."""
    df_data = []
    for result in results:
        if result:
            df_data.append({
                'Filename': result['filename'],
                'Prediction': result['prediction'],
                'Confidence (%)': f"{result['confidence']:.2f}",
                'Normal Prob (%)': f"{result['prob_normal']:.2f}",
                'TB Prob (%)': f"{result['prob_tb']:.2f}"
            })
    
    return pd.DataFrame(df_data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("<h1 class='header-title'>🫁 Tuberculosis Detection System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading AI models..."):
        svm_model, nb_model = load_models()
    
    if svm_model is None or nb_model is None:
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    #Sidebar
    st.sidebar.markdown("""
        <div style="background-color: #ffebee; padding: 1rem; border-radius: 5px; border-left: 4px solid #d32f2f; margin-bottom: 1rem;">
            <p style="color: #c62828; margin: 0; font-weight: 500;">
                ⚠️ This application is for educational purposes only. Not for clinical use.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Select AI Model:",
        ["SVM (Hierarchical)", "Naive Bayes (Hierarchical)"],
        help="Choose which machine learning model to use for prediction"
    )
    
    selected_model = svm_model if "SVM" in model_choice else nb_model
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**📋 Instructions:**\n\n"
        "1. Choose your AI model\n"
        "2. Upload single image or multiple images\n"
        "3. Or upload a ZIP file containing images\n"
        "4. Click 'Predict' to analyze\n"
        "5. View detailed results below"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**🎯 Model Info:**\n\n"
        f"Currently using: **{model_choice}**\n\n"
        "- SVM: Support Vector Machine\n"
        "- NB: Naive Bayes\n"
        "- Both use hierarchical classification"
    )
    
    # Main Content
    st.header("Upload Chest X-Ray Images", "")
    
    # Create tabs for different upload methods
    tab1, tab2 = st.tabs(["Single/Multiple Files", "ZIP Folder"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            # Preview uploaded images
            if st.checkbox("Preview uploaded images", value=True):
                # Gunakan 6 kolom agar preview lebih kecil
                cols = st.columns(6)
                for idx, uploaded_file in enumerate(uploaded_files[:6]):
                    with cols[idx % 6]:
                        image = Image.open(uploaded_file)
                        # Resize image untuk thumbnail preview
                        image.thumbnail((150, 150))
                        st.image(image, caption=uploaded_file.name, use_container_width=True)
                
                if len(uploaded_files) > 6:
                    st.info(f"Showing 6 of {len(uploaded_files)} images. All will be processed.")
            
            # Predict Button
            if st.button("🔬 Start Prediction", type="primary",use_container_width=True):
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    image = Image.open(uploaded_file)
                    result = predict_image(image, selected_model, uploaded_file.name)
                    
                    if result:
                        results.append(result)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                # Display Results
                st.markdown("---")
                st.header("Prediction Results", 'font size=32px;')
                
                # Summary Statistics
                if results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_images = len(results)
                    normal_count = sum(1 for r in results if r['prediction'] == 'Normal')
                    tb_count = sum(1 for r in results if r['prediction'] == 'Tuberculosis')
                    avg_confidence = np.mean([r['confidence'] for r in results])
                    
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Total Images", total_images)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Normal", normal_count, delta=f"{normal_count/total_images*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("TB Detected", tb_count, delta=f"{tb_count/total_images*100:.1f}%", delta_color="inverse")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Results Table
                    st.subheader("📋 Summary Table")
                    results_df = create_results_dataframe(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download Results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name=f"tb_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Detailed Results for Each Image
                    st.subheader("🔍 Detailed Analysis")
                    
                    for idx, result in enumerate(results):
                        with st.expander(f"📄 {result['filename']} - {result['prediction']}", expanded=(idx==0)):
                            # Result Card
                            card_class = "normal-result" if result['prediction'] == 'Normal' else "tb-result"
                            
                            st.markdown(f"""
                                <div class="result-card {card_class}">
                                    <h3>{'✅ NORMAL' if result['prediction'] == 'Normal' else '⚠️ TUBERCULOSIS DETECTED'}</h3>
                                    <p><strong>Confidence:</strong> {result['confidence']:.2f}%</p>
                                    <p><strong>Normal Probability:</strong> {result['prob_normal']:.2f}%</p>
                                    <p><strong>TB Probability:</strong> {result['prob_tb']:.2f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Processed Images
                            img_col1, img_col2, img_col3 = st.columns(3)
                            
                            with img_col1:
                                st.image(result['original'], caption="Original (Resized)", use_container_width=True, clamp=True)
                            
                            with img_col2:
                                st.image(result['enhanced'], caption="Enhanced (CLAHE)", use_container_width=True, clamp=True)
                            
                            with img_col3:
                                st.image(result['mask'], caption="Lung Mask", use_container_width=True, clamp=True)
                            
    
    with tab2:
        uploaded_zip = st.file_uploader(
            "Choose a ZIP file",
            type=["zip"],
            label_visibility="collapsed"
        )
        
        if uploaded_zip:
            st.success(f"✅ ZIP file uploaded: {uploaded_zip.name}")
            
            if st.button("🔬 Extract & Predict", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract ZIP
                    with st.spinner("Extracting ZIP file..."):
                        zip_path = os.path.join(temp_dir, uploaded_zip.name)
                        with open(zip_path, 'wb') as f:
                            f.write(uploaded_zip.getbuffer())
                        
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                    
                    # Find all images
                    image_files = extract_images_from_folder(temp_dir)
                    
                    if not image_files:
                        st.error("❌ No image files found in the ZIP!")
                        st.stop()
                    
                    st.info(f"Found {len(image_files)} images in ZIP file")
                    
                    # Process images
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, img_path in enumerate(image_files):
                        filename = os.path.basename(img_path)
                        status_text.text(f"Processing {idx+1}/{len(image_files)}: {filename}")
                        
                        try:
                            image = Image.open(img_path)
                            result = predict_image(image, selected_model, filename)
                            
                            if result:
                                results.append(result)
                        except Exception as e:
                            st.warning(f"⚠️ Could not process {filename}: {e}")
                        
                        progress_bar.progress((idx + 1) / len(image_files))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display Results (same as tab1)
                    if results:
                        st.markdown("---")
                        st.header("📊 Prediction Results")
                        
                        # Summary Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_images = len(results)
                        normal_count = sum(1 for r in results if r['prediction'] == 'Normal')
                        tb_count = sum(1 for r in results if r['prediction'] == 'Tuberculosis')
                        avg_confidence = np.mean([r['confidence'] for r in results])
                        
                        with col1:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Total Images", total_images)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Normal", normal_count, delta=f"{normal_count/total_images*100:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("TB Detected", tb_count, delta=f"{tb_count/total_images*100:.1f}%", delta_color="inverse")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Results Table
                        st.subheader("Summary Table")
                        results_df = create_results_dataframe(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download Results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv,
                            file_name=f"tb_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🏥 <strong>TB Detection System</strong> | Digital Image Processing Project</p>
            <p><small>Powered by Machine Learning & Computer Vision</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
