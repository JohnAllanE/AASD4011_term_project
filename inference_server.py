
from flask import Flask, request, jsonify

# Uncomment and use these as needed for real model inference
# from tensorflow.keras.models import load_model
# import numpy as np
# import pickle
# import shared_project_functions as spf

app = Flask(__name__)

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json()
    input_text = data.get('text', '').split(' ')
    # TODO: Replace with real model inference
    suggestions = [input_text[0],input_text[1]] if len(input_text)>=2 else ["no suggestions"]
    print(f"Input text: {input_text}, Length: {len(input_text)}, Suggestions: {suggestions}")
    return jsonify({"suggestions": suggestions})

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)  