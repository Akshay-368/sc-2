# Spam Message Classifier

A simple Streamlit app to classify messages as spam or ham using a Naive Bayes model.

## Getting Started

### Prerequisites

- Python 3.x
- All required Python packages are listed in `requirements.txt`.

### Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/spam-message-classifier.git
    cd spam-message-classifier
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the model (optional)**:
    - If you need to retrain the model, run:
    ```bash
    python train_and_save_model.py
    ```

    This will save the model and vectorizer as `.pkl` files.

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

5. **Usage**:
    - Enter a message in the input box to see if it’s classified as spam or ham.

### Files

- `app.py`: The Streamlit app to classify messages.
- `train_and_save_model.py`: Script to train and save the model.
- `vectorizer.pkl` and `spam_classifier.pkl`: Saved vectorizer and model files (generated by `train_and_save_model.py`).

## License

This project is licensed under the MIT License.