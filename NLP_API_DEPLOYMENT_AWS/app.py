import json
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load model and tokenizer
model_path = "bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def lambda_handler(event, context):
    try:
        # Parse input data
        body = json.loads(event["body"])
        text = body.get("text", "")

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted label
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": predicted_label})
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
