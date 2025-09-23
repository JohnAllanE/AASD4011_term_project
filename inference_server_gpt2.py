from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPT2Tokenizer
import os

HOST = "0.0.0.0"; PORT = 6000

app = Flask(__name__)


@app.route("/gpt2", methods=["POST"])
def gpt2():
    j = request.get_json(force=True) or {}
    input_text = j.get("input_text","")
    save_dir = j.get("save_dir", "")
    
    generator = pipeline(
        "text-generation",
        model=save_dir,
        tokenizer=save_dir,
        pad_token_id=GPT2Tokenizer.from_pretrained("gpt2").eos_token_id
    )

    suggestions = generator(input_text, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return jsonify({"suggestions":[suggestions]})

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
