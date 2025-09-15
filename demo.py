import streamlit as st
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="Email Demo", layout="centered")

to = st.text_input("To")
subject = st.text_input("Subject")

st.markdown("**Body:**")
st.text_area("Body", key="body", height=200)

st.markdown("""
<div class="toolbar">
    <button disabled>B</button>
    <button disabled>I</button>
    <button disabled>U</button>
    <button disabled>&#128206;</button>
    <button disabled>&#128247;</button>
</div>
""", unsafe_allow_html=True)