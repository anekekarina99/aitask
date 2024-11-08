
# Trash Classification with TensorFlow ResNet

This repository contains a ResNet-based model for classifying images of trash into categories. The project utilizes TensorFlow, with data augmentation, model tracking through WandB, and automation via GitHub Actions. The trained model is published on Hugging Face Hub.

## Project Structure

- `notebooks/`: Jupyter Notebook with the training pipeline.
- `src/`: Python scripts for data preprocessing, model training, and evaluation.
- `README.md`: Instructions for setup and usage.
- `requirements.txt`: List of dependencies.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/trash-classification.git
cd trash-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Set up your WandB API key and authenticate:

```bash
export WANDB_API_KEY="your_wandb_api_key"
```

### 4. Run the Notebook

Open and execute the pipeline in `notebooks/Trash_Classification.ipynb`:

```bash
jupyter notebook notebooks/Trash_Classification.ipynb
```

### 5. Run Training with Scripts

Alternatively, use the following command to start training:

```bash
python src/train_model.py
```

## Model Training & Saving

The model will be saved as `simpan_resnet_model.h5` if it meets accuracy and stability thresholds (80% accuracy, max 5% train-validation gap).

## Publishing on Hugging Face

Upload to Hugging Face Hub using `huggingface-cli`:

```bash
huggingface-cli login
huggingface-cli upload simpan_resnet_model.h5 --repo your_hf_repo_name
```

## Running Predictions

Run predictions with `src/predict_image.py` by providing an image path:

```bash
python src/predict_image.py --img_path path/to/your/image.jpg
```

## Automation with GitHub Actions

This project includes a GitHub Actions workflow that automates model training. Simply push to start the workflow.

## Results and Evaluation

Evaluate the model on test data:

```bash
python src/evaluate_model.py
```

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the creators of the TrashNet dataset and the TensorFlow and Hugging Face teams.
