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
        font-size: 1.3em !important;
    }}
    h1, .stMarkdown h1, .stHeading, .stMarkdownContainer h1 {{
        color: #222E3C !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1,5])
with col1:
    st.markdown('<div style="text-align: right; color: #222E3C; font-size: 1.4em; font-weight: 500;">To:</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 22px"></div>', unsafe_allow_html=True)  # Spacer
    st.markdown('<div style="text-align: right; color: #222E3C; font-size: 1.4em; font-weight: 500;">Subject:</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 20px"></div>', unsafe_allow_html=True)  # Spacer
    st.markdown('<div style="text-align: right; color: #222E3C; font-size: 1.4em; font-weight: 500;">Body:</div>', unsafe_allow_html=True)

selected_model = "shakespeare"  # Placeholder for model selection
model_list = ['initial']  # Placeholder for model list

# Call inference server for autocomplete suggestions
# Return JSON with suggestions and model list
def get_suggestions(text) -> dict:
    try:
        response = requests.post(
            "http://localhost:5000/autocomplete",
            json={"model": selected_model,
                  "text": text},
            timeout=2
        )
        if response.status_code == 200: # Successful response
            return {"suggestions": response.json().get("suggestions", []),
                    "model_list": response.json().get("model_list", [])
            }
        else:
            return {"suggestions": ["error1"], "model_list": ["error1"]}
    except Exception as e:
        return {"suggestions": ["error2"], "model_list": ["error2"]}

# Get current text from the custom component (initially empty)
if 'body_text' not in st.session_state:
    st.session_state['body_text'] = ""

#Initialize model list
@st.cache_data
def get_models_from_server():
    response = get_suggestions("init")  # Empty text to get model list
    model_list = response.get("model_list", ['error in get_model_from_server'])
    return model_list
model_list = get_models_from_server()

# Get suggestions from inference server based on current text
suggestions = []
if st.session_state['body_text']:
    # get last sentence of body text only
    last_sentence = st.session_state['body_text'].strip().split('.')[-1]  # [-1]: last sentence
    suggestions_response = get_suggestions(last_sentence)
    print(f"DEBUG: suggestions from server: {suggestions_response}")
    suggestion_list = suggestions_response.get('suggestions', []) if isinstance(suggestions_response, dict) else []
    suggestion_to_show = suggestion_list[0] if suggestion_list else ""
else:
    suggestion_to_show = ""

# Optionally display all suggestions below
if suggestions:
    st.markdown("<b>Suggestions:</b>", unsafe_allow_html=True)
    for s in suggestions:
        st.markdown(f"- {s}")
with col2:
    to = st.text_input("", key="to", label_visibility="collapsed")
    subject = st.text_input("", key="subject", label_visibility="collapsed")
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