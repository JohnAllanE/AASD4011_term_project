import os
import streamlit.components.v1 as components

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend/build")
_component_func = components.declare_component("text_monitor", path=build_dir)

def text_monitor(
    default="",
    suggestion="",
    input_color="#0057b8",
    suggestion_color="#b22222",
    spacing_color="#fff",
    font_size="1.2em",
    box_bg_color="#fafaf0",
    key=None,
    height=152
):
    return _component_func(
        default=default,
        suggestion=suggestion,
        input_color=input_color,
        suggestion_color=suggestion_color,
        spacing_color=spacing_color,
        font_size=font_size,
        box_bg_color=box_bg_color,
        key=key,
        height=height
    )