import os
# Disable optimizations for TensorFlow that might cause issues with PyTorch
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging

# Suppress unnecessary logs from Hugging Face
logging.set_verbosity_error()


# Define the model to use for text generation
model_to_use = "gpt2-xl"

# Load the tokenizer and model for text generation
tokenizer = AutoTokenizer.from_pretrained(model_to_use)
model = AutoModelForCausalLM.from_pretrained(model_to_use)

# Move the model to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load a pre-trained safety classifier model to detect hate speech in generated text
safety_checker = pipeline("text-classification", model = "facebook/roberta-hate-speech-dynabench-r4-target")


# Define the input text for the model to generate from
input_text = (
    "The scholar moved deeper into the catacombs, the flickering torchlight casting long shadows on the damp stone walls. "
    "Each step echoed like a whisper of the past, warning of the secrets buried beneath Eldoria. "
    "A brittle manuscript lay atop an altar, its words faded with age—yet their meaning was clear: some knowledge should remain lost."
)

# Tokenize the input text and prepare it for the model
inputs = tokenizer(input_text, return_tensors = "pt").to(device)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate text using the model with specified sampling techniques for diverse output
generated_ids = model.generate(
    input_ids,  # Tokenized input IDs
    attention_mask = attention_mask,  # Attention mask to indicate important tokens
    num_return_sequences = 1,  # Generate only one sequence
    do_sample = True,  # Use sampling instead of greedy decoding
    top_k = 50,  # Limit sampling to the top 50 tokens
    top_p = 0.8,  # Apply nucleus sampling (keep 80% cumulative probability)
    pad_token_id = tokenizer.eos_token_id,  # Use EOS token for padding
    eos_token_id = tokenizer.eos_token_id,  # End-of-sequence token ID
    max_new_tokens = 250,  # Limit the number of new tokens generated
    repetition_penalty = 1.5,  # Penalize repetition in the generated text
    num_beams = 3,  # Number of beams for beam search (helps with output quality)
    early_stopping = True  # Stop generation early when EOS token is reached
)

# Decode the generated tokens back to text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens = True)


# Run safety check to ensure the generated text is safe (no hate speech)
safety_result = safety_checker(generated_text)[0]

# Print the generated text if it's deemed safe
if safety_result["label"] != "hate_speech":

    print(generated_text)

else:

    print("Generated text was flagged as unsafe and was not displayed.")