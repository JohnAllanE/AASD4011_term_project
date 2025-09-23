# Transformer

You will need model_0_shakespeare folder and shakespeare_transformer folder to run [MathProj_GPT2_GROUP.ipynb](./MathProj_GPT2_GROUP.ipynb) or the demo. They are uploaded to the project google drive.

## Running the demo

To run the demo you will need three different python instances, each with its own environment, and they communicate through networking magic.

### Launch Procedures

You will need THREE terminal consoles, one for each environment. The following procedure should be follow IN ORDER.

(Note: [build the environments first!!!](#environments))

### Terminal 0 - Transformer

``` sh
# Initiate the environment (Linux, and Mac-os)
source .venv_transformer/bin/activate

# Run the transformer server
python inference_server_gpt2.py
```

### Terminal 1 - Tensorflow

``` sh
# Initiate the environment (Linux, and Mac-os)
source .venv_tensorflow/bin/activate

# Run the transformer server
python inference_server.py
```

It is important you see the following output before proceding to the next step.

``` console
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.2.19:5000
Press CTRL+C to quit
```

### Terminal 2 - Streamlit Demo

``` sh
# Initiate the environment (Linux, and Mac-os)
source .venv_demo/bin/activate

# Run the streamlit demo, it should open a browser window
streamlit run demo.py
```


## Environments

This environment works for me, may not work for you. Blame Python, not me.

### Transformer Environment

``` sh
# Create a new virtual environment
python3 -m venv .venv_transformer
```

``` sh
# Initiate the environment (Linux, and Mac-os)
source .venv_transformer/bin/activate
```

``` sh
# Install all the required packages
pip install -r requirements_transformer.txt
```

``` sh
# Install en_core_web_sm
python -m spacy download en_core_web_sm
```

### Tensorflow Environment

``` sh
# Create a new virtual environment
python3 -m venv .venv_tensorflow
```

``` sh
# Initiate the environment (Linux, and Mac-os)
source .venv_tensorflow/bin/activate
```

``` sh
# Install all the required packages
pip install -r requirements_tensorflow.txt
```

### Demo Environment

``` sh
# Create a new virtual environment
python3 -m venv .venv_demo
```

``` sh
# Initiate the environment (Linux, and Mac-os)
source .venv_demo/bin/activate
```

``` sh
# Install all the required packages
pip install -r requirements_streamlit.txt
```

You should also install Node.js and npm for custom streamlit components. Please search how to install these for your operating system.
