Diabetes Prediction Using Naive Bayes
Welcome to the Diabetes Prediction project repository! This project uses the Naive Bayes algorithm to predict the likelihood of a person having diabetes based on medical data.

Features
Data Preprocessing: Clean and prepare the dataset for analysis.
Model Training: Train a Naive Bayes classifier on the dataset.
Model Evaluation: Evaluate the performance of the model using various metrics.
Prediction: Use the trained model to make predictions on new data.
Technologies Used
Programming Language: Python
Libraries:
Pandas
NumPy
Scikit-learn
Matplotlib (for visualizations)
Dataset
The dataset used in this project is the PIMA Indians Diabetes Dataset.

Getting Started
Prerequisites
Python 3.x
pip
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Project
Ensure the dataset is in the project directory or update the path in the script accordingly.
Run the Jupyter Notebook:
bash
Copy code
jupyter notebook
Open diabetes_prediction.ipynb and run all cells to see the data preprocessing, model training, and evaluation steps.
Usage
Training the Model
The Jupyter Notebook includes the following steps:

Load and explore the dataset.
Preprocess the data (handle missing values, feature scaling, etc.).
Split the data into training and testing sets.
Train a Naive Bayes classifier.
Evaluate the model's performance.
Making Predictions
After training the model, you can use it to make predictions on new data:

Load the new data.
Preprocess the new data similarly to the training data.
Use the trained model to predict the likelihood of diabetes.
Example:

python
Copy code
new_data = [[5, 166, 72, 19, 175, 25.8, 0.587, 51]]
prediction = model.predict(new_data)
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
Results
The performance of the Naive Bayes classifier is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrix and ROC curve visualizations are also provided.
