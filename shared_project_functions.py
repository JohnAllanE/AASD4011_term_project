from tensorflow.keras.models import load_model
import numpy as np
import json
import pickle

def get_target_subdirectory(corpus_name: str, subdir_string: str = "model"):
    import os

    #first, find if there is already a subdirectory ending in corpus_name
    for dir in os.listdir():
        if os.path.isdir(dir) and dir.endswith('_' + corpus_name):
            return dir
        
    #get index number of already-existing model directories
    model_numbers = []
    for dir in os.listdir():
        if os.path.isdir(dir) and dir.startswith(f"{subdir_string}_"):
            n = dir.replace(f"{subdir_string}_", "").split("_")[0]
            model_numbers.append(int(n))

    first_unused = 0
    while first_unused in model_numbers:
        first_unused += 1
    print(f"/{subdir_string}_{first_unused}_{corpus_name}/")
    
    #create subdirectory
    dir = f"{subdir_string}_{first_unused}_{corpus_name}"
    os.makedirs(dir, exist_ok=True)

    # Return the model number
    return dir

def load_trained_model_and_data(base_argument):
    # Load the trained Keras model from the .keras file
    model = load_model(f'{base_argument}_model.keras')

    # Load the vocabulary mappings
    with open(f'{base_argument}_word_to_id.json', 'r') as f:
        word_to_id = json.load(f)

    # The id_to_word dictionary can be created from word_to_id
    id_to_word = {v: k for k, v in word_to_id.items()}

    # Load max_seq_length from pkl file
    preprocessed_data = pickle.load(open(f'{base_argument}_preprocessed_data.pkl', 'rb'))
    max_seq_length = preprocessed_data['max_seq_length']
    
    return {"model": model, 
            "word_to_id": word_to_id, 
            "id_to_word": id_to_word, 
            "max_seq_length": max_seq_length}

def generate_text(model, seed_text, num_words_to_generate, word_to_id, id_to_word, max_seq_length):
    # Pre-process the seed text
    # It must be in the same format as your training data: lowercase and numerical
    processed_seed = [word_to_id.get(word.lower(), word_to_id['<UNK>']) for word in seed_text.split()]

    generated_text = "" #seed_text

    for _ in range(num_words_to_generate):
        # Pad the input sequence to the required length
        padded_sequence = np.array(processed_seed[-max_seq_length:] + [word_to_id['<PAD>']] * (max_seq_length - len(processed_seed[-max_seq_length:])))
        
        # Reshape for the model's input layer
        padded_sequence = padded_sequence.reshape(1, max_seq_length)
        
        # Get the model's prediction
        # The output is a probability distribution over the entire vocabulary
        predictions = model.predict(padded_sequence, verbose=0)[0]
        
        # Sample the next word from the distribution
        # This is a key step to make the output more varied and creative
        predicted_id = np.random.choice(len(predictions), p=predictions)
        
        # Convert the integer ID back to a word
        predicted_word = id_to_word[predicted_id]
        
        # Stop generation if the end-of-sentence token is predicted
        if predicted_word == '<EOS>':
            break
            
        generated_text += " " + predicted_word
        processed_seed.append(predicted_id)

    return generated_text