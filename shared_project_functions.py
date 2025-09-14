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