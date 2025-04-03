


from flask import Flask, render_template, request
import pickle
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

app = Flask(__name__, template_folder="templates")

# Load the transformer model
transformer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Class Labels
class_labels = {
    0: "Not_offensive",
    1: "Offensive_Untargeted",
    2: "Offensive_target_insult_Group",
    3: "Offensive_target_insult_individual",
    4: "not-malayalam",
    5: "unknown"
}

# Load models for Classifier-based predictions
models = {
    "Logistic Regression": pickle.load(open("logistic_Regression.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
    "Naïve Bayes": pickle.load(open("Naive_Bayes.pkl", "rb")),
    "KNN": pickle.load(open("KNN.pkl", "rb")),
}

# Define MLP model architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MLP Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = MLPClassifier(768, len(class_labels)).to(device)
state_dict = torch.load("MLP_model1.pkl", map_location=device)
mlp_model.load_state_dict(state_dict, strict=False)
mlp_model.eval()

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Classifier-based Model Page
@app.route('/classifiers', methods=['GET', 'POST'])
def classifiers():
    if request.method == 'POST':
        comment = request.form['comment']
        comment_vector = transformer.encode([comment])
        predictions = {name: model.predict(comment_vector)[0] for name, model in models.items()}
        best_prediction = Counter(predictions.values()).most_common(1)[0][0]
        best_label = class_labels.get(best_prediction, "Unknown")
        return render_template('classifiers.html', user_comment=comment, prediction_text=f'Best Prediction: {best_label}')
    
    return render_template('classifiers.html', user_comment="", prediction_text="")

# MLP Model Page
@app.route('/mlp', methods=['GET', 'POST'])
def mlp():
    if request.method == 'POST':
        comment = request.form['comment']
        comment_vector = transformer.encode([comment])
        comment_tensor = torch.tensor(comment_vector, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = mlp_model(comment_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        prediction_label = class_labels.get(prediction, "Unknown")
        return render_template('mlp.html', user_comment=comment, prediction_text=f'Prediction: {prediction_label}')
    
    return render_template('mlp.html', user_comment="", prediction_text="")

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Get the assigned port from Render
    app.run(host="0.0.0.0", port=port, debug=True)