Here’s an updated `pytorch-llm/README.md` with a new **PyInstaller section** covering installation, building standalone macOS applications, packaging the model for text generation, and troubleshooting.



# PyTorch LLM

A small GPT-style language model implemented from scratch using PyTorch. This project demonstrates tokenization, embeddings, causal self-attention, transformer blocks, training, checkpointing, and autoregressive text generation.

## Features

- Character-level tokenization
- Token and positional embeddings
- Multi-head causal self-attention
- Feed-forward neural networks
- Transformer blocks with residual connections and layer normalization
- GPT-style language model
- Training with PyTorch and AdamW
- Model checkpoint saving and loading
- Autoregressive text generation
- Unit tests
- Optional packaging with PyInstaller

## Project Structure

```text
pytorch-llm/
├── config.py
├── train.py
├── generate.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── .gitkeep
│   ├── sample.txt
│   └── processed/
├── tokenizer/
│   ├── __init__.py
│   └── tokenizer.py
├── model/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── transformer.py
│   └── gpt.py
├── training/
│   ├── __init__.py
│   ├── dataset.py
│   ├── trainer.py
│   └── checkpoint.py
├── checkpoints/
└── tests/
    ├── test_tokenizer.py
    ├── test_attention.py
    └── test_model.py
```

## Requirements

- Python 3.10 or newer
- PyTorch
- pip
- macOS, Linux, or Windows

On Apple Silicon Macs, PyTorch may use Apple's Metal Performance Shaders (MPS) backend when supported.

## Installation

Clone or download the project, then open Terminal and navigate to the project directory:

```bash
cd pytorch-llm
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you intend to package the project as an executable, install PyInstaller too:

```bash
pip install pyinstaller
```

## Prepare Training Data

Place your training text in:

```text
data/sample.txt
```

You can use your own plain-text corpus. The character-level tokenizer builds its vocabulary from the training text, so the vocabulary determines which characters the model can process.

Open `config.py` and set the data path:

```python
data_path: str = "data/sample.txt"
```

The default configuration uses a context length of 128 tokens and a batch size of 32. Your training corpus must be sufficiently large for the configured context length, training/validation split, and batch size.

For a small demonstration corpus, consider temporarily reducing these settings:

```python
block_size: int = 32
batch_size: int = 8
n_embd: int = 64
n_head: int = 4
n_layer: int = 2
```

The context length, embedding size, and number of attention heads must be configured consistently.

## Train the Model

From the project root, run:

```bash
python train.py
```

The training process:

1. Reads the training text.
2. Builds a character-level vocabulary.
3. Encodes the text into token IDs.
4. Creates training and validation datasets.
5. Initializes the GPT model.
6. Trains the model using AdamW.
7. Evaluates training and validation loss.
8. Saves model checkpoints.

Checkpoints are written to the `checkpoints/` directory.

Depending on your training implementation, the project may produce files such as:

```text
checkpoints/
├── model.pt
└── model_best.pt
```

A small dataset is useful for testing the pipeline, but a model trained on a small corpus will generally produce limited or repetitive text.

## Generate Text

After training, run:

```bash
python generate.py
```

Enter a prompt when requested. The model will generate text based on the prompt and its learned vocabulary.

Enter `quit` to exit the interactive session.

**Important:** The generation script must load the same tokenizer vocabulary and compatible model configuration used during training. A prompt containing characters outside the saved vocabulary cannot be encoded by the character-level tokenizer.

## Run the Tests

Run the complete test suite from the project root:

```bash
python -m unittest discover -s tests -v
```

You can also run individual test modules:

```bash
python -m unittest tests.test_tokenizer -v
python -m unittest tests.test_attention -v
python -m unittest tests.test_model -v
```

If Python cannot import a test module, ensure `tests/__init__.py` exists.

## Configuration

The main settings are in `config.py`.

| Setting | Description |
|---|---|
| `block_size` | Maximum context length |
| `vocab_size` | Number of unique tokens |
| `n_embd` | Embedding dimension |
| `n_head` | Number of attention heads |
| `n_layer` | Number of transformer blocks |
| `dropout` | Dropout probability |
| `batch_size` | Training batch size |
| `max_iters` | Number of training iterations |
| `learning_rate` | Optimizer learning rate |
| `eval_interval` | Frequency of training evaluation |
| `data_path` | Path to the training text |
| `checkpoint_path` | Path to the model checkpoint |

## Package the Model with PyInstaller

[PyInstaller](https://pyinstaller.org/) packages a Python application and its dependencies into a distributable application directory or executable.

This project can use PyInstaller to package the text-generation script. The trained model checkpoint remains a separate file, allowing you to distribute or replace the model without rebuilding the application.

### 1. Install PyInstaller

Activate your project's virtual environment:

```bash
source .venv/bin/activate
```

Install PyInstaller:

```bash
python -m pip install pyinstaller
```

Verify the installation:

```bash
pyinstaller --version
```

### 2. Build the Text-Generation Application

From the project root, run:

```bash
pyinstaller --clean --onedir --name pytorch-llm generate.py
```

PyInstaller creates the following directories and files:

```text
pytorch-llm/
├── build/
├── dist/
│   └── pytorch-llm/
│       ├── pytorch-llm
│       └── ...
└── pytorch-llm.spec
```

The `dist/pytorch-llm/` directory contains the packaged application and its dependencies.

The `--onedir` option is a good starting point for PyTorch projects because it produces an application directory rather than one large self-extracting executable.

### 3. Run the Packaged Application

On macOS, run the generated executable from Terminal:

```bash
./dist/pytorch-llm/pytorch-llm
```

The application may need the model checkpoint at the location expected by `generate.py`.

For example, if the script loads:

```text
checkpoints/model.pt
```

make sure the checkpoint is available at that relative path from the application's working directory, or modify the script to accept an explicit checkpoint path.

A convenient distribution layout is:

```text
pytorch-llm-release/
├── pytorch-llm/
│   ├── pytorch-llm
│   └── ...
└── checkpoints/
    └── model.pt
```

The precise layout depends on how `generate.py` resolves checkpoint paths. Relative paths are generally resolved from the process's current working directory, not automatically from the executable's directory.

### 4. Include the Checkpoint in the Package

For a fixed model, you can bundle the checkpoint with the application:

```bash
pyinstaller \
  --clean \
  --onedir \
  --name pytorch-llm \
  --add-data "checkpoints/model.pt:checkpoints" \
  generate.py
```

On macOS, the `--add-data` separator is a colon (`:`).

If the checkpoint is missing when you build the application, train the model first.

**Important:** Including the checkpoint in the application does not automatically make your script locate it correctly. Your Python code must resolve the bundled file's path appropriately. If you want users to replace the model, keeping the checkpoint external is usually more convenient.

PyTorch is a large dependency, so packaged applications may require substantial disk space.

### 5. Build a Single-File Executable

If you prefer a single executable, use:

```bash
pyinstaller \
  --clean \
  --onefile \
  --name pytorch-llm \
  generate.py
```

The resulting executable is located at:

```text
dist/pytorch-llm
```

Run it with:

```bash
./dist/pytorch-llm
```

Single-file packaging can increase startup time because bundled files may need to be extracted at launch. With PyTorch, `--onedir` is often easier to troubleshoot.

### 6. Package the Training Script

You can also package the training entry point:

```bash
pyinstaller \
  --clean \
  --onedir \
  --name pytorch-llm-train \
  train.py
```

The packaged training application still needs access to its training corpus. Make sure `data/sample.txt`, or whichever file is configured in `config.py`, is available at the path the script expects.

For most development workflows, running `python train.py` directly is simpler. PyInstaller is particularly useful when you want to distribute a ready-to-run application to someone who does not have Python installed.

### 7. Important macOS Notes

- **Build on the target operating system.** PyInstaller is not a general cross-compiler. Build a macOS application on macOS.
- **Match the processor architecture.** An application built in an Apple Silicon Python environment generally targets Apple Silicon. Building for Intel Macs may require a separate Intel-compatible Python environment and dependencies.
- **Test on a clean environment.** Run the packaged application without relying on files or Python packages that exist only in your development environment.
- **Expect a large package.** PyTorch and its native libraries can significantly increase application size.
- **Check security restrictions.** Applications distributed to other Mac users may require code signing and notarization, depending on how they are distributed.
- **Use a terminal during development.** It makes missing dependencies, invalid paths, and checkpoint-loading errors easier to diagnose.

## Troubleshooting

### Not enough training data

If the dataset is too small, training or validation loaders may not contain enough sequences or complete batches.

Try adding more text or temporarily reducing `block_size` and `batch_size`.

### Unknown character during generation

The character-level tokenizer only recognizes characters included in the training vocabulary.

Use a prompt composed of known characters, or retrain the model with a corpus containing the required characters.

### Checkpoint not found

Verify that the checkpoint exists and that `checkpoint_path` points to the correct file.

For a packaged application, remember that the current working directory may differ from the project directory.

### PyInstaller reports missing modules

Try rebuilding with a clean build directory:

```bash
pyinstaller --clean --onedir --name pytorch-llm generate.py
```

For dependencies that PyInstaller cannot detect automatically, inspect its warnings and use the appropriate hidden-import or collection options. Avoid collecting every dependency indiscriminately unless necessary, since this can make the package much larger.

### PyTorch or MPS is unavailable

The packaged application can only use backends supported by the installed PyTorch build and the machine running it. Test CPU execution first if accelerator initialization fails.

## Learning Goals

This project is intended to help you understand:

- How text becomes token IDs
- How embeddings represent tokens and positions
- How causal self-attention works
- How transformer blocks combine attention and feed-forward layers
- How autoregressive language models learn to predict tokens
- How inference differs from training
- How model checkpoints preserve trained weights and tokenizer mappings
- How Python ML applications can be packaged for distribution

## Future Improvements

Potential extensions include:

- Byte-pair encoding (BPE) or another subword tokenizer
- Key-value caching for faster autoregressive generation
- Learning-rate scheduling
- Mixed-precision training
- Gradient accumulation
- Better evaluation metrics
- A command-line interface for generation
- A graphical user interface
- Quantization for reduced memory use
- Disaggregated prefill and decode serving

## Disclaimer

This is an educational implementation intended to demonstrate the basic components of a GPT-style language model. It is not a production-ready inference engine.
```

### After updating the README

Save the content as `README.md` in your project root. Then, from Terminal, test the PyInstaller workflow:

```bash
cd pytorch-llm
source .venv/bin/activate
pip install pyinstaller
python train.py
pyinstaller --clean --onedir --name pytorch-llm generate.py
```

**One important detail:** PyInstaller packages your Python code and dependencies, but your generation script still needs to find the trained checkpoint and load the saved tokenizer vocabulary. The README documents both options: keeping the checkpoint external or bundling it with the application.
