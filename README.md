NL2SQL: Natural Language to SQL Converter
This project implements a system to convert natural language questions into SQL queries using a fine-tuned T5 model. It leverages the WikiSQL dataset for training and provides a user-friendly Gradio interface for inference. The code supports training, saving/loading model files (weights, config, tokenizer), and generating SQL queries from text input.
Features

Model Training: Fine-tunes a T5-small model on the WikiSQL dataset to translate natural language to SQL.
Inference: Generates SQL Queries from natural language questions using a trained model.
Gradio Interface: Provides an interactive web UI for easy query generation.
Model Persistence: Saves and loads model weights, configuration, and tokenizer for future use.
GPU/CPU Support: Automatically detects and uses CUDA if available.

Prerequisites

Python 3.8 or higher
CUDA-enabled GPU (optional, for faster training/inference)
Git (for cloning the repository)

Installation

Clone the Repository:
git clone https://github.com/your-username/nl2sql.git
cd nl2sql


Create a Virtual Environment (recommended):
python -m venv nl2sql_env
source nl2sql_env/bin/activate  # On Windows: nl2sql_env\Scripts\activate


Install Dependencies:
pip install torch torchvision transformers datasets gradio tqdm

If using a GPU, ensure torch is installed with the correct CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version



Usage
Training the Model
To train the model (or if no saved model exists), run the script with training enabled:
python nl2sql_train_and_infer.py


The script will check for existing model files in nl2sql_t5_model/. If none are found or training is forced, it trains a T5-small model on the WikiSQL dataset for 3 epochs.
Trained model files (weights, config, tokenizer) are saved in nl2sql_t5_model/.

Running Inference with Gradio
To use the pre-trained model for inference:
python nl2sql_train_and_infer.py


If model files exist, the script loads them and launches a Gradio interface in your default web browser.
Enter a natural language question (e.g., "Show all employees with salary above 50000") in the textbox.
The generated SQL query will appear in the output section.

To force training even if model files exist, modify the script's main call:
create_gradio_interface(train_model_flag=True)

Saved Files
The following files are saved in nl2sql_t5_model/ after training:

model.pt: PyTorch model weights
config.json: Model configuration
spiece.model: Tokenizer vocabulary
tokenizer.json: Tokenizer configuration
special_tokens_map.json: Special tokens configuration
added_tokens.json: Additional tokens (if any)

Project Structure
nl2sql/
├── nl2sql_train_and_infer.py  # Main script for training and inference
├── nl2sql_t5_model/           # Directory for saved model files
├── README.md                  # Project documentation

Example
Input Question: "List all products with price greater than 100"Output SQL: SELECT * FROM products WHERE price > 100
Troubleshooting

Version Conflicts: Ensure compatible versions of torch, torchvision, and transformers. Update with:pip install --upgrade torch torchvision transformers


CUDA Issues: Verify your CUDA version and install the matching torch version.
Gradio Not Loading: Check if port 7860 is free, or specify a different port:iface.launch(server_port=8080)



Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with Hugging Face Transformers and Gradio.
Trained on the WikiSQL dataset.

