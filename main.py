import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# Upload the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    init_image = Image.open(uploaded_file).convert("RGB")
    st.image(init_image, caption="Original Image", use_column_width=True)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to(device)

    # Define prompt and generate
    prompt = "A high-resolution, studio-quality image of a full frontal view of a fashion garment on a clean white background..."
    strength = 0.5

    with torch.autocast(device):
        output = pipe(prompt=prompt, image=init_image, strength=strength).images[0]

    st.image(output, caption="Refined Image", use_column_width=True)
