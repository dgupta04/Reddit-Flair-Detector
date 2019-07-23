from flask import Flask, render_template, request
from worker import predictor

app = Flask(__name__)

@app.route('/')
def make():
    return render_template('index.html')

@app.route('/getFlair', methods=['POST', 'GET'])
def predict():
    if request.method=='POST':
        x = request.form['postURL']
        flairName = predictor.predictFlair(x)
        return render_template('flair.html', flair=flairName)

if __name__ == '__main__':
    app.run(debug=True, port=33507)