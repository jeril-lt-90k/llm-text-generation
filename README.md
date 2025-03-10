
# Text Generation with Safety Filtering

This project demonstrates how to use a language model (GPT-2, specifically `gpt2-xl`) to generate text from a given input prompt. Additionally, it includes a safety filter that checks the generated text for hate speech before displaying it, ensuring the output is safe and appropriate.

## Features

- **Text Generation**: Leverages GPT-2 (`gpt2-xl` by default) to generate creative and coherent text based on a user-provided prompt.
- **Safety Filtering**: Uses a pre-trained model (`facebook/roberta-hate-speech-dynabench-r4-target`) to detect and filter out hate speech in the generated text.
- **GPU Support**: The code automatically uses a GPU if available, otherwise falls back to CPU for processing.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face `transformers` library
- `facebook/roberta-hate-speech-dynabench-r4-target` model for hate speech detection
- CUDA (optional, for GPU support)

## Installation

1. Clone the repository or download the code.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a CUDA-compatible GPU and necessary drivers installed for GPU acceleration. If you don't have a GPU, the script will automatically run on CPU.

## How to Use

1. Clone the repository or download the script.
2. Install the dependencies (as described above).
3. Run the script to generate text and perform a safety check:

   ```bash
   python generate_text.py
   ```

   You can modify the `input_text` variable within the script to experiment with different input prompts for text generation.

### Example Input & Output

For example, when the input is:

```
"The scholar moved deeper into the catacombs, the flickering torchlight casting long shadows on the damp stone walls..."
```

The generated text might be:

```
"The scholar ventured deeper into the catacombs, his steps echoing in the silence, with only the distant sound of dripping water as company."
```

If any generated text is flagged for containing hate speech, the message will appear as:

```
Generated text was flagged as unsafe and was not displayed.
```

## License

This project uses various models and libraries, including:

- **GPT-2** from Hugging Face's `transformers` library (MIT License).
- **Hate Speech Detection Model** from Facebook's `roberta-hate-speech-dynabench-r4-target` (MIT License).

For more details, please refer to the respective libraries and models' documentation.

## Attribution

- **GPT-2 model**: The GPT-2 model used for text generation is part of the Hugging Face Transformers library. You can find more details and the model's license [here](https://huggingface.co/).
  
- **Hate Speech Detection Model**: The safety filter used in this project, the `roberta-hate-speech-dynabench-r4-target`, is from Facebook AI. You can find more information and the model's license [here](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target).

