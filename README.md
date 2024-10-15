# Large-Language-Model
I designed my own Large Language Model and tokenizer in Python using Pytorch and Numpy
- My Large-Language Model takes in a line seed text as input and generates a piece of synthetic, continuation text based on a set of training data as specified in input.txt
- I implemented the Byte Pair Encoding (BPE) algorithm for the tokenizer to tokenize input text data and training data
- Implemented a Generative Pre-trained Transformer(GPT) model using Pytorch with CUDA, encompassing multiple layers of self-attention mechanisms, positional encodings, and position-wise feedforward networks
- Previous inputs and generation is automatically fed back into the GPT as context
- The final model is outputted to a .pth file for reusability
