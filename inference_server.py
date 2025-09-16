
from flask import Flask, request, jsonify

from tensorflow.keras.models import load_model
import numpy as np
import pickle
import shared_project_functions as spf

model_list = {
    "shakespeare": "model_0_shakespeare/shakespeare",
    "lotr": None
}

model_data = {"shakespeare": spf.load_trained_model_and_data(model_list["shakespeare"]), 
          "lotr": None}

app = Flask(__name__)

last_input_text = None
suggestions = []
model_name = None

@app.route('/autocomplete', methods=['POST'])
def autocomplete():     
    
    #Only proceed if the request is different from the last one
    #Store generated text in global variables
    global last_input_text
    global suggestions
    global model_name
    
    
    data = request.get_json()
    input_text = data.get('text', 'default1')
    model_name = data.get('model', 'default2')

    if input_text != last_input_text:
        last_input_text = input_text # Update the last input text
        print("New request received") 
        words = input_text.split(' ') #split into words
        #print(f"Model: {model_name}, Input text: {input_text}, Words: {words}")
        # TODO: Replace with real model inference
        if not words or not model_name:
            suggestions = ["no suggestions"]
        else:
            # Use the selected model to generate suggestions
            if model_name in model_data and model_data[model_name] is not None:
                model_info = model_data[model_name]
                model = model_info["model"]
                word_to_id = model_info["word_to_id"]
                id_to_word = model_info["id_to_word"]
                max_seq_length = model_info["max_seq_length"]
                # Generate text using the model
                num_words_to_generate = 5  # Number of words to generate
                generated_text = spf.generate_text(model, input_text, num_words_to_generate, word_to_id, id_to_word, max_seq_length)
                words = generated_text.split(' ')
                
                # Return only the newly generated words as suggestions
                suggestions = [generated_text]
            else:
                suggestions = ["model not found"]
        print(f"Input text: {input_text}, Length: {len(input_text)}, Suggestions: {suggestions}")
    else:
        print("Duplicate request, using cached suggestions")
    return jsonify({"suggestions": suggestions,
                    "model_list": list(model_list.keys())})

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)