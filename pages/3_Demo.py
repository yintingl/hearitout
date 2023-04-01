import streamlit as st

from PIL import Image

image = Image.open('demo.png')

st.image(image, caption='GPT powered summary and Q&A')