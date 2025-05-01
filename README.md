# Encoder-Decoder Architecture in PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A clean, modular implementation of an encoder-decoder architecture in PyTorch with comprehensive visualizations. This project demonstrates sequence-to-sequence modeling with GRU (Gated Recurrent Unit) networks.

![Encoder-Decoder Architecture](https://via.placeholder.com/800x400?text=Encoder-Decoder+Architecture)

## 🌟 Features

- **Modular Architecture**: Separate Encoder and Decoder classes that can be easily customized
- **Visualization Tools**: Comprehensive set of visualization functions for model understanding
- **Training Pipeline**: Complete training and evaluation workflow
- **Attention Visualization**: Visual representation of attention weights
- **Hidden State Analysis**: Tools to inspect model internals during inference

## 📋 Requirements

```
torch>=1.9.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
```

## 🚀 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/encoder-decoder-pytorch.git

# Navigate to the project directory
cd encoder-decoder-pytorch

# Install dependencies
pip install -r requirements.txt
```

## 🔍 Project Structure

```
encoder-decoder-pytorch/
├── main.py               # Main script to run the model
├── models/
│   ├── encoder.py        # Encoder implementation
│   ├── decoder.py        # Decoder implementation
│   └── seq2seq.py        # Combined encoder-decoder model
├── utils/
│   ├── dataset.py        # Dataset creation and processing
│   └── visualization.py  # Visualization functions
├── notebooks/
│   └── examples.ipynb    # Example usage in notebook format
└── README.md
```

## 🧠 How It Works

The encoder-decoder architecture works by:

1. **Encoding Phase**: The encoder processes the input sequence and compresses all information into a context vector (hidden state)
2. **Decoding Phase**: The decoder takes the context vector and generates the output sequence step by step

This implementation uses GRU units for both encoder and decoder, with options for attention mechanisms.

![Hidden State Visualization](https://via.placeholder.com/800x400?text=Hidden+State+Visualization)

## 🛠️ Usage

### Basic Example

```python
import torch
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import EncoderDecoder

# Initialize models
input_dim = 1
output_dim = 1
hidden_dim = 64
n_layers = 2

encoder = Encoder(input_dim, hidden_dim, n_layers)
decoder = Decoder(output_dim, hidden_dim, n_layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderDecoder(encoder, decoder, device)

# Create sample data
from utils.dataset import SimpleAdditionDataset
dataset = SimpleAdditionDataset(num_samples=1000, seq_len=10)

# Train model
from main import train
model, losses = train(model, dataset, epochs=30, batch_size=32)

# Visualize results
from utils.visualization import predict_and_visualize
test_input, test_target = dataset[0]
predict_and_visualize(model, test_input, test_target)
```

### Visualization Examples

```python
from utils.visualization import (
    visualize_architecture,
    visualize_sample,
    visualize_hidden_states,
    visualize_attention,
    visualize_training_progress
)

# Visualize model architecture
visualize_architecture()

# Visualize sample data
visualize_sample(dataset, idx=0)

# Visualize training progress
visualize_training_progress(losses)
```

## 📈 Results

The model successfully learns to predict cumulative sums in the example task:

![Training Progress](https://via.placeholder.com/800x400?text=Training+Progress+Graph)

## 🔄 Extending the Model

This implementation can be extended for various sequence-to-sequence tasks:

- **Machine Translation**: Process text in one language and output another
- **Text Summarization**: Compress longer texts into shorter summaries
- **Time Series Forecasting**: Predict future values based on past observations
- **Image Captioning**: Generate textual descriptions of images (with additional components)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- PyTorch team for the amazing deep learning framework
- The deep learning community for their continuous research
- All contributors who help improve this project
