import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# App Title
st.title("💬 AI Chatbot using DialoGPT")
st.write("Chat in real-time with an AI trained using NLP and transformers!")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

if "history" not in st.session_state:
    st.session_state.history = None
if "step" not in st.session_state:
    st.session_state.step = 0

# Input Box
user_input = st.text_input("You:", "")

# Chat Logic
if user_input:
    new_input_ids = tokenizer.encode
