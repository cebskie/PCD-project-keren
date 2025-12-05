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
warnings.filterwarnings('ignore')

@st.cache_resource
def load_models(base_path='.'):
    """Load trained models from disk."""
    svm_path = os.path.join(base_path, 'svm_model.pkl')
    nb_path = os.path.join(base_path, 'nb_model.pkl')
    
    svm_model = None
    nb_model = None
    
    try:
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                svm_model = pickle.load(f)
            print(f"✅ SVM model loaded from: {svm_path}")
        else:
            print(f"⚠️ SVM model not found at: {svm_path}")
        
        if os.path.exists(nb_path):
            with open(nb_path, 'rb') as f:
                nb_model = pickle.load(f)
            print(f"✅ Naive Bayes model loaded from: {nb_path}")
        else:
            print(f"⚠️ Naive Bayes model not found at: {nb_path}")
            
        return svm_model, nb_model
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_image(img_path, show_steps=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None

    # Resize to standard size (512x512)
    img = cv2.resize(img, (512, 512))

    # Gaussian filtering to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return img, blurred, enhanced

def segment_lungs(enhanced_img, show_steps=False):
    if enhanced_img is None:
        return None, None, None

    # CRITICAL: Inverse Otsu to get lungs (dark areas in X-ray)
    _, binary = cv2.threshold(enhanced_img, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Morphological closing to fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # Find contours and select largest regions (lungs)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask
    mask = np.zeros_like(enhanced_img)

    if len(contours) > 0:
        # Sort by area and take top 2 (left and right lung)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        num_lungs = min(2, len(contours))

        for i in range(num_lungs):
            if cv2.contourArea(contours[i]) > 1000:  # Filter small regions
                cv2.drawContours(mask, [contours[i]], -1, 255, -1)

    # Final closing on mask
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_final, iterations=2)

    return mask, binary, closed

def extract_shape_features(mask):
    """Extract geometrical features from lung region."""
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

    # Aggregate features from all regions
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
    """Extract first-order statistical features."""
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

    # Histogram-based features
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

    # Check if there's meaningful data
    if np.all(masked_img == 0) or len(np.unique(masked_img)) < 2:
        return {k: 0 for k in ['contrast', 'dissimilarity', 'homogeneity',
                               'energy', 'correlation', 'ASM']}

    # Normalize to 8-bit
    img_8bit = cv2.normalize(masked_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Compute GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    try:
        glcm = graycomatrix(img_8bit, distances=distances, angles=angles,
                           levels=256, symmetric=True, normed=True)

        # Extract properties
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

def extract_all_features(img_path):
    # Preprocessing
    original, blurred, enhanced = preprocess_image(img_path, show_steps=False)

    if enhanced is None:
        return None, None, None

    # Segmentation
    lung_mask, binary, closed = segment_lungs(enhanced, show_steps=False)

    if lung_mask is None:
        return None, None, None

    # Feature extraction
    shape_features = extract_shape_features(lung_mask)
    stats_features = extract_first_order_stats(enhanced, lung_mask)
    glcm_features = extract_glcm_features(enhanced, lung_mask)

    # Combine all features
    all_features = {**shape_features, **stats_features, **glcm_features}

    return all_features, lung_mask, enhanced

#STREAMLIT APP
def main():
    st.set_page_config(
        page_title="TB Detection System",
        layout="wide"
    )
    
    st.title("Tuberculosis Detection from Chest X-Ray Images")
    st.markdown("---")

    #Load Models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    svm_model, nb_model = load_models(base_path=current_dir)

    if svm_model is None or nb_model is None:
        st.error("❌ Models could not be loaded. Please train the models first in Training_Model.ipynb")
        st.info(f"Looking for models in: {current_dir}")
        return
    
    # Sidebar for user input
    st.sidebar.header("Configuration User INPUT")
    model_choice = st.sidebar.selectbox(
        "Select Model:",
        ["SVM (Hierarchical)", "Naive Bayes (Hierarchical)"]
    )

    #Upload File
    uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["png", "jpg", "jpeg"])

if __name__ == "__main__":
    main()