# from tensorflow.keras.models import load_model
# import numpy as np

def get_target_subdirectory(corpus_name: str, subdir_string: str = "model"):
    """
    Finds or creates a target subdirectory for a given corpus name and subdirectory prefix.
    This function searches the current working directory for a subdirectory ending with the specified corpus name.
    If such a directory exists, its name is returned. Otherwise, it creates a new subdirectory with a unique
    index and the given corpus name, following the pattern: '{subdir_string}_{index}_{corpus_name}'.
    Args:
        corpus_name (str): The name of the corpus to be used in the subdirectory name.
        subdir_string (str, optional): The prefix for the subdirectory name. Defaults to "model".
    Returns:
        str: The name of the found or newly created subdirectory.
    
    """
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
    """
    Loads a trained Keras model and associated data required for text processing.
    Given a base file path argument, this function loads:
    - A trained Keras model from a `.keras` file.
    - A word-to-ID mapping from a JSON file.
    - An ID-to-word mapping (constructed from the word-to-ID mapping).
    - The maximum sequence length from a pickled preprocessed data file.
    Args:
        base_argument (str): The base file path (without extension) used to locate the model and data files.
    Returns:
        dict: A dictionary containing:
            - "model": The loaded Keras model.
            - "word_to_id": Dictionary mapping words to their integer IDs.
            - "id_to_word": Dictionary mapping integer IDs to words.
            - "max_seq_length": The maximum sequence length used during preprocessing.
    Raises:
        FileNotFoundError: If any of the required files are not found.
        Exception: For errors during model or data loading
    
    """
    
    from tensorflow.keras.models import load_model
    import numpy as np
    import json
    import pickle
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
    import numpy as np
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