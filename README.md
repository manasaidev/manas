# MANAS

![MANAS Logo](logo.jpg)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Twitter API](https://img.shields.io/badge/twitter-api-blue.svg)](https://developer.twitter.com/en/docs)
[![Tensorflow](https://img.shields.io/badge/tensorflow-2.9.1-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.21.1-green.svg)](https://huggingface.co/docs/transformers/index)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
  - [Transformer Model](#transformer-model)
  - [Sentiment Analyzer](#sentiment-analyzer)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Twitter Integration](#twitter-integration)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Project MANAS is an advanced AI model designed to generate custom responses to tweets related to cryptocurrencies. The model leverages state-of-the-art natural language processing techniques, including transformer-based architectures and sentiment analysis, to provide engaging and contextually relevant responses.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/project-manas.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the necessary API credentials:
   - Twitter API: Create a developer account at [https://developer.twitter.com/](https://developer.twitter.com/) and obtain the required credentials (BEARER_TOKEN, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET).
4. Update the `config.py` file with your API credentials.

## Usage
To run the MANAS AI model and start monitoring Twitter for mentions:
```
python main.py
```

The model will continuously monitor for tweets mentioning the specified account (@youraccount) and generate responses based on the trained model.

## Model Architecture
### Transformer Model
The core of the MANAS AI model is a transformer-based architecture implemented using the PyTorch framework. The transformer model consists of an encoder and decoder, along with attention mechanisms and positional encoding. The model is trained on a large corpus of cryptocurrency-related text data to capture the nuances and semantics of the domain.

### Sentiment Analyzer
In addition to the transformer model, MANAS incorporates a sentiment analyzer module that assesses the sentiment of the input text. The sentiment analyzer is built using the BERT (Bidirectional Encoder Representations from Transformers) model, fine-tuned on a sentiment classification task. The sentiment information is used to guide the response generation process and ensure appropriate tone and emotion in the generated responses.

## Data Preprocessing
The training data for the MANAS AI model undergoes extensive preprocessing to ensure high-quality input. The preprocessing steps include:
- Tokenization: The text data is tokenized into individual words or subwords using advanced tokenization techniques.
- Stop word removal: Common stop words that do not contribute to the meaning of the text are removed.
- Lowercasing: All text is converted to lowercase to reduce vocabulary size and improve generalization.
- Special token handling: Special tokens such as `<pad>`, `<unk>`, and `<eos>` are added to handle padding, unknown words, and end-of-sequence markers.

## Training
The MANAS AI model is trained using a combination of the transformer model and sentiment analyzer. The training process involves the following steps:
1. Load the preprocessed training data.
2. Initialize the transformer model and sentiment analyzer.
3. Define the loss function and optimizer.
4. Iterate over the training data for a specified number of epochs.
5. Forward pass: Pass the input through the transformer model and sentiment analyzer.
6. Compute the loss based on the predicted and target sequences.
7. Backward pass: Compute gradients and update model parameters.
8. Monitor the training loss and perplexity.
9. Save the trained model checkpoint.

## Twitter Integration
MANAS seamlessly integrates with the Twitter API to monitor for mentions and generate responses in real-time. The `TwitterClient` class handles the authentication and interaction with the Twitter API. It provides methods to search for mentions and post tweets as replies.

The main loop of the program continuously monitors for mentions of the specified account (@youraccount) and processes each mention by extracting the relevant text, generating a response using the trained model, and posting the response as a reply to the mention.

## Evaluation
The performance of the MANAS AI model can be evaluated using various metrics, such as:
- Perplexity: Measures how well the model predicts the next word in a sequence.
- BLEU score: Evaluates the quality of the generated responses by comparing them to reference responses.
- Sentiment accuracy: Assesses the accuracy of the sentiment analyzer in predicting the sentiment of input text.
- Human evaluation: Manual evaluation of the generated responses by human raters to assess coherence, relevance, and engagement.

## Contributing
Contributions to Project MANAS are welcome! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Submit a pull request detailing your changes.

Please ensure that your contributions adhere to the project's coding style and guidelines.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more information.

---

For more detailed information and advanced usage, please refer to the [documentation](docs/).

If you have any questions or need assistance, please feel free to [open an issue](https://github.com/manasaidev/manas/issues) or contact the project maintainers.