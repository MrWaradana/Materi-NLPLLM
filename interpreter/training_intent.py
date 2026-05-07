import torch
import pandas as pd 
import numpy as np 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

bertModel = 'bert-base-uncased'
bertModel = 'distilbert-base-uncased'

n_epochs = 10

# Load the dataset 

# ====== Dataset ======
data = {
    "text": [
    # greet
    "hallo", "hello", "hi",  "good morning",  
    "hey dude", "good afternoon",

    # goodbye
    "good bye", "cee you later", "byebye","goodbye", "bye bye", "see you later",

    # affirm
    "yes", "yeah", "indeed", "of course", "that sounds good", "correct",

    # deny
    "no", "not", "never", "don't like that", "no way", "not really",

    # mood_great
    "perfect", "great", "amazing", "wonderful",
    "so good", "so perfect",

    # mood_unhappy
    "my day was horrible", "I am sad", "I don't feel very well", "I am disappointed", "I'm so sad",

    # bot_challenge
    "are you a bot?", "are you a human?", "am I talking to a bot?", "am I talking to a human?",

    # provide_name
    "My name is <name>",  "I am <name>",  "It's <name>",   "Name <name>", "It is <name>",

    # provide_email
    "My email is <email>", "This email <email> is mine", "You can reach me at <email>", "email saya adalah <email>",

    # provide_nrp
    "My nim is <nrp>", "My nrp is <nrp>", "NRP saya <nrp>", "NRP <nrp>","NIM <nrp>",


    # user_regist
    "I want to register", "Registration please", "I am a new user",

    # show_info
    "Show me my registration", "What did I give?", "Review my data",
    
    "show my transcript",
    "give me my transcript",
    "show transcript",
    "please show my grades",
    "show the transcript"

    ],
    
    "label": [
    # greet
    "greet","greet","greet","greet","greet","greet", 

    # goodbye
    "goodbye","goodbye","goodbye","goodbye","goodbye","goodbye", 

    # affirm
    "affirm","affirm","affirm","affirm","affirm","affirm",

    # deny
    "deny","deny","deny","deny","deny","deny",

    # mood_great
    "mood_great","mood_great","mood_great","mood_great","mood_great",    "mood_great",

    # mood_unhappy
    "mood_unhappy","mood_unhappy","mood_unhappy","mood_unhappy","mood_unhappy",

    # bot_challenge
    "bot_challenge","bot_challenge","bot_challenge","bot_challenge",

    # provide_name
    "provide_name","provide_name","provide_name","provide_name", "provide_name",

    # provide_email
    "provide_email","provide_email","provide_email","provide_email",

    # provide_nrp
    "provide_nrp","provide_nrp","provide_nrp","provide_nrp", "provide_nrp",


    # user_regist
    "user_regist","user_regist","user_regist",

    # show_info
    "show_info","show_info","show_info",
    
    # minta_transkrip
    "minta_transkrip", "minta_transkrip", "minta_transkrip", "minta_transkrip", "minta_transkrip"
    ]
}
df = pd.DataFrame(data)

texts = list(df["text"])
labels = df["label"].values

categories = np.unique(labels)
sample_size = len(texts)
num_class = len(categories)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.3, random_state=42)

# Tokenizer and Dataset Class
tokenizer = BertTokenizer.from_pretrained(bertModel)

class myDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


print ("Prepare the datasets")

# Prepare the datasets
train_dataset = myDataset(X_train, y_train, tokenizer)
test_dataset = myDataset(X_test, y_test, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


print ("Load the BERT model")

# Load BERT model
model = BertForSequenceClassification.from_pretrained(bertModel, num_labels=num_class)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print ("Device yang dipakai: ", device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()


# save the model
def save_model(model, tokenizer, model_name = "my_intent_model"):
   # ======  Save Model and Tokenizer ======
   model.save_pretrained(model_name)
   tokenizer.save_pretrained(model_name)
   import pickle
   with open(model_name+"/label_encoder.pkl", "wb") as f:
       pickle.dump(label_encoder, f)

   return 

# Training loop
def train_model(model, train_loader, loss_fn, optimizer, device, epochs=n_epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions.double() / len(train_loader.dataset)
        train_losses.append(avg_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy:.4f}')

    return train_losses

# Evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print (all_labels)
    print (all_preds)
    
    print (type(all_labels))
    print (type(all_preds))
        
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(all_labels, all_preds))
    
    return accuracy

# Train the model
train_losses = train_model(model, train_loader, loss_fn, optimizer, device)

save_model(model, tokenizer)

# Evaluate the model
accuracy = evaluate_model(model, test_loader, device)

# Plot training loss
plt.plot(train_losses, label='Training loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




