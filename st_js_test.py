import streamlit as st
from js_text_monitor import text_monitor

st.title("Streamlit Text Monitor Demo")

text = text_monitor()

st.write("Output from app.py:", text)