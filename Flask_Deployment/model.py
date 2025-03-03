import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask=attention_mask).logits

# Load the trained model
def load_model():
    model = SpamClassifier()
    model.load_state_dict(torch.load("spam_classifier.pth", map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

model = load_model()
