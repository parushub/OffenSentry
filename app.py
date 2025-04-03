'''from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']

    # Predict using the model
    prediction = model.predict([comment])[0]  # Ensure the model takes a list

    # Print result in console
    print(f"Comment: {comment} | Prediction: {prediction}")

    return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
'''

# from flask import Flask, render_template, request
# import pickle
# from sentence_transformers import SentenceTransformer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# app = Flask(__name__, template_folder="templates")

# # Load the same transformer model used during training
# transformer = SentenceTransformer("bert-base-multilingual-cased")

# # Define class labels
# class_labels = {
#     0: "Not_offensive",
#     1: "Offensive_Untargeted",
#     2: "Offensive_target_insult_Group",
#     3: "Offensive_target_insult_individual",
#     4: "not-malayalam",
#     5: "unknown"
# }

# # Define the correct MLP model structure (must match saved model)
# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)  # Output layer (logits)
#         return x

# # Load MLP model separately (PyTorch model)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define model with correct architecture
# input_dim = 768  # BERT embedding size
# num_classes = len(class_labels)  # Should be 6 (matches output layer)

# mlp_model = MLPClassifier(input_dim, num_classes).to(device)

# # Load the saved model weights
# state_dict = torch.load("MLP_model1.pkl", map_location=device)

# # Fix layer name mismatch if needed
# for key in list(state_dict.keys()):
#     if key.startswith("module."):
#         state_dict[key[7:]] = state_dict.pop(key)  # Remove "module." prefix

# # Load weights into model
# mlp_model.load_state_dict(state_dict, strict=False)  # Allow missing keys

# mlp_model.eval()  # Set to evaluation mode

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     comment = request.form['comment']

#     # Convert text to numerical representation using the same transformer
#     comment_vector = transformer.encode([comment])

#     # Convert to PyTorch tensor
#     comment_tensor = torch.tensor(comment_vector, dtype=torch.float32).to(device)

#     # Make prediction
#     with torch.no_grad():
#         output = mlp_model(comment_tensor)
#         prediction = torch.argmax(output, dim=1).item()  # Get the class index
    
#     # Get the label name from dictionary
#     prediction_label = class_labels.get(prediction, "Unknown")

#     print(f"Comment: {comment} | Prediction: {prediction} ({prediction_label})")

#     return render_template('index.html', user_comment=comment, prediction_text=f'Prediction: {prediction} ({prediction_label})')

# if __name__ == '__main__':
#     app.run(debug=True)



























# from flask import Flask, render_template, request
# import pickle
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from collections import Counter
# import torch

# app = Flask(__name__, template_folder="templates")

# # Load all trained models
# models = {
#     "Logistic Regression": pickle.load(open("logistic _Regression.pkl", "rb")),
#     # "Linear SVM": pickle.load(open("poly_labse.pkl", "rb")),
#     "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
#     "Naïve Bayes": pickle.load(open("Naive_Bayes.pkl", "rb")),
#     "KNN": pickle.load(open("KNN.pkl", "rb")),
#     # "SVM_Poly": pickle.load(open("svmpoly.pkl", "rb")),
#     # "AdaBoost": pickle.load(open("adaboost.pkl", "rb")),
#     # "SVM_RBF": pickle.load(open("svm_rbf_smote.pkl", "rb")),


# }

# # Load the same transformer model used during training
# transformer = SentenceTransformer("bert-base-multilingual-cased")

# # Define class labels
# class_labels = {
#     0: "Not_offensive",
#     1: "Offensive_Untargeted",
#     2: "Offensive_target_insult_Group",
#     3: "Offensive_target_insult_individual",
#     4: "not-malayalam",
#     5: "unknown"
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     comment = request.form['comment']

#     # Convert text to numerical representation
#     comment_vector = transformer.encode([comment])

#     # Store predictions from all classifiers
#     predictions = {}
    
#     for model_name, model in models.items():
#         pred = model.predict(comment_vector)[0]
#         predictions[model_name] = pred

#     # Print predictions for debugging
#     print(f"Predictions: {predictions}")

#     # Apply **majority voting** to determine the best prediction
#     prediction_counts = Counter(predictions.values())
#     best_prediction = prediction_counts.most_common(1)[0][0]  # Most frequent label

#     # Get the label name
#     best_label = class_labels.get(best_prediction, "Unknown")

#     return render_template(
#         'index.html',
#         user_comment=comment,
#         prediction_text=f'Best Prediction: {best_prediction} ({best_label})'
#     )

# if __name__ == '__main__':
#     app.run(debug=True)













# from flask import Flask, render_template, request
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# import numpy as np

# app = Flask(__name__, template_folder="templates")

# # Load fine-tuned MuRIL model and tokenizer
# # MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/muril_classification"
# MODEL_PATH = "D:\site"
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# # Define class labels
# class_labels = {
#     0: "Not_offensive",
#     1: "Offensive_Untargeted",
#     2: "Offensive_target_insult_Group",
#     3: "Offensive_target_insult_individual",
#     4: "not-malayalam"
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     comment = request.form['comment']

#     # Tokenize the input comment
#     inputs = tokenizer(comment, padding=True, truncation=True, max_length=128, return_tensors="pt")

#     # Perform prediction
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         predicted_class = torch.argmax(predictions, dim=1).item()

#     # Get the label name
#     best_label = class_labels.get(predicted_class, "Unknown")

#     return render_template(
#         'index.html',
#         user_comment=comment,
#         prediction_text=f'Best Prediction: {best_label}'
#     )

# if __name__ == '__main__':
#     app.run(debug=True)













from flask import Flask, render_template, request
import pickle
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

app = Flask(__name__, template_folder="templates")

# Load the transformer model
transformer = SentenceTransformer("bert-base-multilingual-cased")

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
    app.run(debug=True)
