from flask import Flask, request, render_template
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

app = Flask(__name__)

# Load the trained model
pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', SGDClassifier(loss='log_loss', warm_start=True, max_iter=1000, l1_ratio=0.03, penalty='l2', alpha=1e-4, fit_intercept=False))
])

# Assume the model has been trained and serialized to a file 'clickbait_model.pkl'
with open('clickbait_model.pkl', 'rb') as model_file:
    pipe = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        headline = request.form['headline']
        prediction = pipe.predict([headline])
        result = 'Clickbait' if prediction == 1 else 'Not Clickbait'
        return render_template('result.html', headline=headline, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

