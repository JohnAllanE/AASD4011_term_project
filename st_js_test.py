import streamlit as st
from js_text_monitor import text_monitor
#import shared_project_functions as spf
import requests

# --- Use st_autorefresh for periodic polling ---
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1000, key="refresh")

# Set a custom background colour using CSS

page_background_colour = "#F7F9FB"
js_input_colour = "#222E3C"
js_suggestion_colour = "#1976D2"
js_spacing_colour = "#F7F9FB"
js_font_size = "1.4em"
js_box_bg_colour = "#FFFFFF"
toolbar_bg_colour = page_background_colour

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {page_background_colour};
    }}
    label, .stTextInput label, .stTextInput input {{
        color: #222E3C !important;
    }}
    .stTextInput input {{
        background-color: #fff !important;
        color: #222E3C !important;
    }}
    h1, .stMarkdown h1, .stHeading, .stMarkdownContainer h1 {{
        color: #222E3C !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Custom styled labels
st.markdown('<span style="color: #222E3C; font-size: 1.1em; font-weight: 500;">To</span>', unsafe_allow_html=True)
to = st.text_input("", key="to", label_visibility="collapsed")

st.markdown('<span style="color: #222E3C; font-size: 1.1em; font-weight: 500;">Subject</span>', unsafe_allow_html=True)
subject = st.text_input("", key="subject", label_visibility="collapsed")

st.markdown('<span style="color: #222E3C; font-size: 1.1em; font-weight: 500;">Body</span>', unsafe_allow_html=True)

# Call inference server for autocomplete suggestions
def get_suggestions(text):
    try:
        response = requests.post(
            "http://localhost:5000/autocomplete",
            json={"text": text},
            timeout=2
        )
        if response.status_code == 200:
            return response.json().get("suggestions", [])
        else:
            return ["no suggestions-error"]
    except Exception as e:
        return ["no suggestions-exception"]

# Get current text from the custom component (initially empty)
if 'body_text' not in st.session_state:
    st.session_state['body_text'] = ""

# Get suggestions from inference server based on current text
suggestions = []
if st.session_state['body_text']:
    suggestions = get_suggestions(st.session_state['body_text'])
    suggestion_to_show = suggestions[0] if suggestions else ""
else:
    suggestion_to_show = ""

# Show the text monitor with the first suggestion
text = text_monitor(
    suggestion=suggestion_to_show,
    input_color=js_input_colour,
    suggestion_color=js_suggestion_colour,
    spacing_color=js_spacing_colour,
    font_size=js_font_size,
    box_bg_color=js_box_bg_colour,
    height=300,
    key="body_text"
)

# Optionally display all suggestions below
if suggestions:
    st.markdown("<b>Suggestions:</b>", unsafe_allow_html=True)
    for s in suggestions:
        st.markdown(f"- {s}")

print(f"suggestions: {suggestions}")
st.markdown(f"""
<div class="toolbar" style="background: {toolbar_bg_colour}; padding: 8px; border-radius: 6px;">
    <button style="width:40px; height:40px; font-size: 1.2em;">B</button>
    <button style="width:40px; height:40px; font-size: 1.2em;">I</button>
    <button style="width:40px; height:40px; font-size: 1.2em;">U</button>
    <button style="width:40px; height:40px; font-size: 1.2em;">&#10554;</button>
    <button style="width:40px; height:40px; font-size: 1.2em;">&#10555;</button>
    <button style="width:40px; height:40px; font-size: 1.2em;">&#43;</button>
    <button style="width:40px; height:40px; font-size: 1.2em;">&#9786;</button>

</div>
""", unsafe_allow_html=True)