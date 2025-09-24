from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import os

os.environ["TRANSFORMERS_NO_TF"] = "1"

torch_models = {
    "shakespeareX": "model_5_shakespeareX/shakespeareX",
    "aristotleX": "model_6_aristotleX/aristotleX"
}

save_dirs = {
    "shakespeareX": "model_5_shakespeareX/shakespeare_gpt2_final",
    "aristotleX": "model_6_aristotleX/aristotle_trained_gpt2_model"
}

tokenizers = {
    "shakespeareX": GPT2Tokenizer.from_pretrained(save_dirs["shakespeareX"]),
    "aristotleX": GPT2Tokenizer.from_pretrained(save_dirs["aristotleX"])
}

models = {
    "shakespeareX": GPT2LMHeadModel.from_pretrained(save_dirs["shakespeareX"]),
    "aristotleX": GPT2LMHeadModel.from_pretrained(save_dirs["aristotleX"])
}

#save_dir = "model_5_shakespeareX/shakespeare_gpt2_final"
#tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
#model = GPT2LMHeadModel.from_pretrained(save_dir)

generators = {
    "shakespeareX": pipeline(
        "text-generation",
        model=models["shakespeareX"],
        tokenizer=tokenizers["shakespeareX"],
        pad_token_id=tokenizers["shakespeareX"].eos_token_id
    ),
    "aristotleX": pipeline(
        "text-generation",
        model=models["aristotleX"],
        tokenizer=tokenizers["aristotleX"],
        pad_token_id=tokenizers["aristotleX"].eos_token_id
    )
}

# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     pad_token_id=tokenizer.eos_token_id
# )

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    model_name = data.get("model", "shakespeareX")
    seed_text = data.get("text", "")
    max_length = int(data.get("max_length", 100))
    num_return_sequences = int(data.get("num_return_sequences", 1))
    try:
        outputs = generators[model_name](
            seed_text,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
        # Return all generated texts as a list
        suggestions = [out["generated_text"] for out in outputs]
        #Split to words
        words = suggestions[0].split(' ')
        #Return only the new words after the seed text
        new_words = words[len(seed_text.split(' ')):]
        suggestion_to_return = ' '.join(new_words)
        suggestions = [suggestion_to_return]
        #DEBUG: print the model and suggestions
        print(f"DEBUG: Model: {model_name}, Generated suggestions: {suggestions}")
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)