from mb_nlp.modernbert_multilabel.model import ModernBertMultiLabelClassifier
import pandas as pd

def main():
    # Use the provided synthetic_train.csv for demonstration
    csv_path = 'tests/synthetic_train.csv'
    text_col = 'text'
    label_cols = ['label']  # Only one label column in this file

    classifier = ModernBertMultiLabelClassifier(model_name='answerdotai/ModernBERT-base', num_labels=1)
    classifier.train_from_csv(csv_path, text_col=text_col, label_cols=label_cols, epochs=1, batch_size=2)

    # Load the texts for prediction
    df = pd.read_csv(csv_path)
    texts = df[text_col].tolist()
    predictions = classifier.predict(texts)
    for text, pred in zip(texts, predictions):
        print(f"Input: {text} => Prediction: {pred}")

if __name__ == "__main__":
    main()
