from mb_bert_classifier import BertTextClassifier

def main():
    classifier = BertTextClassifier(model_name='bert-base-uncased', num_labels=2)
    examples = [
        "This is a great product!",
        "Terrible experience. Will not buy again.",
        "",
        "!!!??",
        "The quick brown fox jumps over the lazy dog. " * 50,  # Long input
        ["Short text", "Another example", ""]  # Batch input
    ]

    print("Single input predictions:")
    for text in examples[:-1]:
        print(f"Input: {repr(text)} => Prediction: {classifier.predict(text)}")

    print("\nBatch input predictions:")
    print(f"Inputs: {examples[-1]} => Predictions: {classifier.predict(examples[-1])}")

if __name__ == "__main__":
    main()
