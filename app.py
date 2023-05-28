from flask import Flask, render_template, request, jsonify

import model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/submit', methods=["POST"])
def detect():
    file=request.files['file']
    file.save('uploaded.png')
    return model.run_model('uploaded.png')

if __name__ == "__main__":
    app.run()