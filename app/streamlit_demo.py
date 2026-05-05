"""Streamlit web demo for the Pokemon classifier project."""



from __future__ import annotations



import sys

from pathlib import Path



import matplotlib.pyplot as plt

import streamlit as st

import torch

from PIL import Image



# Make the src package importable when running from the app directory.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:

    sys.path.insert(0, str(SRC_ROOT))



from data_loader import get_data_transforms

from models import AlexNetClassifier, GoogleNetClassifier, ResNetClassifier, VGGNetClassifier



st.set_page_config(

    page_title="Pokemon Classifier Demo",

    page_icon="🔍",

    layout="wide",

    initial_sidebar_state="expanded",

)



st.markdown(

    """

    <style>

    .main {

        background: linear-gradient(180deg, #f7f8fc 0%, #ffffff 100%);

    }

    .hero {

        padding: 1.2rem 1.4rem;

        border-radius: 18px;

        background: linear-gradient(135deg, #14213d 0%, #1d3557 52%, #457b9d 100%);

        color: white;

        box-shadow: 0 12px 30px rgba(20, 33, 61, 0.16);

    }

    .hero h1 {

        margin: 0;

        font-size: 2rem;

        line-height: 1.1;

    }

    .hero p {

        margin: 0.55rem 0 0;

        opacity: 0.92;

        font-size: 1rem;

    }

    .card {

        background: white;

        border: 1px solid rgba(20, 33, 61, 0.08);

        border-radius: 16px;

        padding: 1rem 1.1rem;

        box-shadow: 0 8px 24px rgba(20, 33, 61, 0.05);

    }

    .muted {

        color: #52616b;

        font-size: 0.95rem;

    }

    </style>

    """,

    unsafe_allow_html=True,

)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path("results")

DATA_ROOT = Path("data/raw")



MODEL_OPTIONS = {

    "ResNet": {

        "factory": ResNetClassifier,

        "checkpoint": RESULTS_DIR / "resnet_best.pth",

        "description": "ResNet50 backbone",

        "default": True,

    },

    "AlexNet": {

        "factory": AlexNetClassifier,

        "checkpoint": RESULTS_DIR / "alexnet_best.pth",

        "description": "AlexNet backbone",

    },

    "VGGNet": {

        "factory": VGGNetClassifier,

        "checkpoint": RESULTS_DIR / "vggnet_best.pth",

        "description": "VGG16 backbone",

    },

    "GoogleNet": {

        "factory": GoogleNetClassifier,

        "checkpoint": RESULTS_DIR / "googlenet_best.pth",

        "description": "GoogleNet backbone",

    },

}





def _class_names_from_disk() -> list[str]:

    pokemon_root = DATA_ROOT / "PokemonData"

    if pokemon_root.exists():

        search_root = pokemon_root

    else:

        search_root = DATA_ROOT



    if not search_root.exists():

        return []



    return [item.name for item in sorted(search_root.iterdir()) if item.is_dir()]





@st.cache_data(show_spinner=False)

def load_class_names() -> list[str]:

    class_names = _class_names_from_disk()

    if class_names:

        return class_names

    return [f"Pokemon_{index}" for index in range(150)]





@st.cache_resource(show_spinner=False)

def load_model(model_name: str, num_classes: int):

    model_info = MODEL_OPTIONS[model_name]

    factory = model_info["factory"]



    if model_name == "VGGNet":

        model = factory(num_classes=num_classes, model_name="vgg16", pretrained=False)

    else:

        model = factory(num_classes=num_classes, pretrained=False)



    checkpoint_path = model_info["checkpoint"]

    loaded_from = None



    if checkpoint_path.exists():

        try:

            state_dict = torch.load(checkpoint_path, map_location=DEVICE)

            model.load_state_dict(state_dict, strict=True)

            loaded_from = str(checkpoint_path)

        except Exception:

            loaded_from = None



    model.to(DEVICE)

    model.eval()

    return model, loaded_from





@st.cache_resource(show_spinner=False)

def get_inference_transform():

    return get_data_transforms(augment=False)["val"]





def predict_topk(image: Image.Image, model, class_names: list[str], top_k: int = 5):

    transform = get_inference_transform()

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)



    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)

        top_probabilities, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[1]))



    predictions = []

    for rank, (probability, index) in enumerate(zip(top_probabilities[0], top_indices[0]), start=1):

        class_index = index.item()

        class_name = class_names[class_index] if class_index < len(class_names) else f"Unknown_{class_index}"

        predictions.append(

            {

                "rank": rank,

                "class_name": class_name,

                "probability": float(probability.item()),

            }

        )



    return predictions





def render_metrics(predictions):

    if not predictions:

        return



    top_prediction = predictions[0]

    metric_columns = st.columns(3)

    metric_columns[0].metric("Top Prediction", top_prediction["class_name"])

    metric_columns[1].metric("Confidence", f"{top_prediction['probability'] * 100:.2f}%")

    metric_columns[2].metric("Predictions", f"Top {len(predictions)}")





def render_probability_chart(predictions):

    if not predictions:

        return



    names = [item["class_name"] for item in predictions]

    probabilities = [item["probability"] for item in predictions]



    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(names, probabilities, color="#457b9d")

    ax.set_xlim(0, 1)

    ax.set_xlabel("Probability")

    ax.set_title("Top Predictions")

    ax.invert_yaxis()



    for bar, probability in zip(bars, probabilities):

        ax.text(probability + 0.01, bar.get_y() + bar.get_height() / 2, f"{probability * 100:.2f}%", va="center")



    st.pyplot(fig, clear_figure=True)





def main():

    class_names = load_class_names()

    num_classes = len(class_names)



    st.markdown(

        """

        <div class="hero">

            <h1>Pokemon Classifier Demo</h1>

            <p>Upload a Pokemon image, choose a backbone, and view the top predictions.</p>

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.write("")



    st.sidebar.markdown("### Controls")

    model_name = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)

    st.sidebar.caption(MODEL_OPTIONS[model_name]["description"])



    try:

        model, loaded_from = load_model(model_name, num_classes)

        model_error = None

    except Exception as error:

        model = None

        loaded_from = None

        model_error = error



    st.sidebar.markdown("---")

    st.sidebar.markdown("### Status")

    st.sidebar.write(f"Device: {DEVICE}")

    st.sidebar.write(f"Classes: {num_classes}")

    if model_error is not None:

        st.sidebar.error(f"Model load failed: {model_error}")

    elif loaded_from:

        st.sidebar.success(f"Loaded checkpoint: {loaded_from}")

    else:

        st.sidebar.warning("Checkpoint not found. Using initialized weights.")



    left_column, right_column = st.columns([1.1, 0.9], gap="large")



    with left_column:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("1. Upload Image")

        uploaded_file = st.file_uploader("Choose a Pokemon image", type=["jpg", "jpeg", "png", "bmp", "gif"])

        st.caption("The demo uses the validation transform from the training pipeline.")



        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(image, use_container_width=True, caption="Uploaded image")

        else:

            image = None

            st.info("Upload an image to run inference.")

        st.markdown("</div>", unsafe_allow_html=True)



    with right_column:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("2. Prediction")



        if model_error is not None:

            st.error("Model could not be loaded. Check the checkpoint file and model definition.")

        elif image is not None and model is not None:

            predictions = predict_topk(image, model, class_names, top_k=5)

            render_metrics(predictions)

            st.write("")

            render_probability_chart(predictions)



            st.markdown("#### Ranked Results")

            for item in predictions:

                st.write(f"{item['rank']}. {item['class_name']} - {item['probability'] * 100:.2f}%")

        else:

            st.write("No image uploaded yet.")

            st.caption("Once you upload an image, the top-5 predictions will appear here.")

        st.markdown("</div>", unsafe_allow_html=True)



    st.write("")

    footer_left, footer_mid, footer_right = st.columns(3)

    footer_left.info(f"Model: {model_name}")

    footer_mid.info(f"Framework: PyTorch")

    footer_right.info(f"Checkpoint: {'Yes' if loaded_from else 'No'}")





if __name__ == "__main__":

    main()