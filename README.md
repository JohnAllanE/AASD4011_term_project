# Transformer Only

This doc is only meant for [MathProj_GPT2_GROUP.ipynb](./MathProj_GPT2_GROUP.ipynb).

You will need model_0_shakespeare folder and shakespeare_transformer folder to run the code. They are uploaded to the project google drive.

## Environment

This environment works for me, may not work for you. Blame Python, not me.

``` console
# # Create a new virtual environment
$ python3 -m venv .venv
```

``` console
# # Initiate the environment (Linux, and Mac I think)
$ source .venv/bin/activate
```

``` console
# # Install all the required packages
$ pip install -r requirements_transformer.txt
```

``` console
# # Install en_core_web_sm
$ python -m spacy download en_core_web_sm
```
