"""Streamlit web interface for Pokemon classifier demo"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import ResNetClassifier, AlexNetClassifier, VGGNetClassifier, GoogleNetClassifier
from data_loader import get_data_transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Page configuration
st.set_page_config(
    page_title="Pokemon Classifier Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .title {
        text-align: center;
        color: #FF0000;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name, num_classes=150):
    """Load model from checkpoint"""
    try:
        if model_name == 'AlexNet':
            model = AlexNetClassifier(num_classes=num_classes, pretrained=False)
        elif model_name == 'VGGNet':
            model = VGGNetClassifier(num_classes=num_classes, model_name='vgg16', pretrained=False)
        elif model_name == 'GoogleNet':
            model = GoogleNetClassifier(num_classes=num_classes, pretrained=False)
        elif model_name == 'ResNet':
            model = ResNetClassifier(num_classes=num_classes, pretrained=False)
        else:
            return None
        
        # Load checkpoint
        checkpoint_path = f'results/{model_name.lower()}_best.pth'
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            return model
        else:
            st.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_class_names():
    """Load class names"""
    data_root = Path('data/raw/PokemonData')
    if not data_root.exists():
        data_root = Path('data/raw')

    class_names = [path.name for path in sorted(data_root.iterdir()) if path.is_dir()]
    if class_names:
        return class_names

    return [f"Pokemon_{i}" for i in range(150)]


def predict_pokemon(image, model, class_names, transforms):
    """Predict Pokemon from image"""
    try:
        # Preprocess image
        image_tensor = transforms(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.topk(probabilities, 5)
        
        # Get results
        predictions = []
        for i in range(5):
            idx = top_idx[0][i].item()
            prob = top_prob[0][i].item()
            predictions.append({
                'rank': i + 1,
                'name': class_names[idx] if idx < len(class_names) else f"Unknown_{idx}",
                'probability': prob
            })
        
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("<h1 class='title'>🔴 Poké Classifier 🔵</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered Pokemon Image Classification</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ['AlexNet', 'VGGNet', 'GoogleNet', 'ResNet'],
        help="Choose which model to use for prediction"
    )
    
    # Load model
    st.sidebar.info(f"Loading {model_choice}...")
    model = load_model(model_choice)
    
    if model is None:
        st.error("❌ Failed to load model. Make sure the checkpoint exists.")
        return
    
    st.sidebar.success(f"✅ {model_choice} loaded successfully!")
    
    # Load class names and transforms
    class_names = load_class_names()
    transforms = get_data_transforms()['val']
    
    # Image input
    st.header("📸 Upload Pokemon Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
            help="Upload a Pokemon image"
        )
    
    with col2:
        use_example = st.checkbox("Use example image")
    
    image_to_predict = None
    
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file).convert('RGB')
    elif use_example:
        # Create a sample image if no file uploaded
        st.info("Please upload an image or select example mode")
    
    # Prediction
    if image_to_predict is not None:
        st.markdown("---")
        
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image_to_predict, use_column_width=True)
        
        # Make prediction
        with col2:
            st.subheader("🎯 Prediction Results")
            
            with st.spinner("Analyzing image..."):
                predictions = predict_pokemon(image_to_predict, model, class_names, transforms)
            
            if predictions:
                # Top prediction
                top_pred = predictions[0]
                st.success(f"**Top Prediction:** {top_pred['name']}")
                st.metric("Confidence", f"{top_pred['probability']*100:.2f}%")
                
                # Top 5 predictions
                st.subheader("Top 5 Predictions")
                for pred in predictions:
                    st.write(
                        f"{pred['rank']}. **{pred['name']}** - "
                        f"{pred['probability']*100:.2f}%"
                    )
                
                # Visualization
                st.subheader("Confidence Distribution")
                names = [p['name'] for p in predictions]
                probs = [p['probability'] for p in predictions]
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(names, probs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
                ax.set_xlabel('Probability')
                ax.set_title('Top 5 Pokemon Predictions')
                ax.set_xlim(0, 1)
                
                # Add percentage labels
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    ax.text(prob, i, f' {prob*100:.2f}%', va='center')
                
                st.pyplot(fig)
    
    # Info section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Model:** " + model_choice)
    
    with col2:
        st.info(f"**Classes:** 150 Pokemon")
    
    with col3:
        st.info("**Framework:** PyTorch")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🎮 Pokemon Classifier v1.0 | Created for educational purposes</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
