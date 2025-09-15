import os
import streamlit.components.v1 as components

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend/build")
_component_func = components.declare_component("text_monitor", path=build_dir)

def text_monitor(default="", key=None, height=100):
    """Streamlit Text Monitor component: returns the live value of the input box."""
    return _component_func(default=default, key=key, height=height)