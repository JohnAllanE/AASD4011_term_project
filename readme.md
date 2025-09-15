
# Project development

## Topic

- Considered style transfer and text generation
- Decided on text generation, with different models built on different styles

## Styles chosen

1. Shakespeare
2. Continue adding here!

## Style ideas

- Single character (Hamlet)
- KJ Bible, Old Testament.
- US Presidents Inaugural Addresses (NLTK download).
- T-Swift Lyrics.
- Monty Python.

## Demo ideas

1. Autocomplete with "style" options
    - UI: Email window with "style" along with formatting options
2. Link two words with generated text
    - not sure if possible)

## To-Do list

- Create "lite model" that won't take 15 hours to train
  - Less training data?
  - Fewer neurons?
  - Fewer layers?
- Try tweaking models to improve result
- Try training with other hardware?
  - Local GPU
  - Google Colab
- Demo part - prototype with Streamlit with existing model
  - Add in additional or different models later
- Alternatives
  - Preprocessing: use TensorFlow Tokenization instead of spaCy
- Documenting / reporting
- Metrics / assessment
  - Accuracy? Confusion matrix? Others?
- Distribute work for project (loosely)

## Things people are working on

- F: styles, lite model, try training on own hardware?
- A: working on DL1 instead, come up with styles
- JA: cleaning up and sharing existing code, demo
- K:
- J
- H:

- Everyone:
  - Create and tweak models
  - Explore mappings and try and explain model issues

## Meeting plans

- Check in on Tuesday after class

## Code information

### Environment

- (JA) Everything so far has been tested with python3.11
  - Later versions may work, not sure - edit this if it works for you
- To create venv from provided requirements file saved in project

```python
pip install -r requirements.txt
```

- Do not share your .venv file (add to .gitignore), everyone should build their own environment
