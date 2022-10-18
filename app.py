from flask import Flask, redirect, url_for, request, render_template
from model import RcmSys
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/training', methods=['POST'])
def training():
    return render_template("training.html")

@app.route('/prediction', methods=['POST'])
def prediction():
    return render_template("prediction.html")

@app.route('/train', methods=['POST'])
def train():
    file_path = request.form['file_path']
    item_col = request.form['item_col']
    user_col = request.form['user_col']
    popularity = eval(request.form['popularity'])
    genre = eval(request.form['genre'])

    global rcmsys
    rcmsys = RcmSys(popularity, genre)
    rcmsys.fit(file_path, item_col, user_col)

    return render_template("training.html", training_text="training is done", prediction="True")

@app.route('/predict', methods=['POST'])
def predict():
    target_user = int(request.form['target_user'])
    threshold = int(request.form['threshold'])
    n = int(request.form['n'])

    # try:
    item_rank = rcmsys.predict(target_user, threshold, n)
    # except NameError:
    # rcmsys = pickle.load(open("model.pkl", 'rb'))
    # item_rank = rcmsys.predict(target_user, threshold, n)
    
    items = item_rank.head(10).reset_index()

    return render_template("prediction.html",  tables=[items.to_html(classes='data', header="true", index=False)])






if __name__ == "__main__":
    app.run(debug=True)