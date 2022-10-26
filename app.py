from flask import Flask, render_template, request, redirect, url_for, flash, json
from models import Modelnv

app = Flask(__name__)  # creating the Flask class object


def CekTingkat(tingkat):
    if tingkat:
        if tingkat == '0':
            label = 'loma'
        elif tingkat == '1':
            label = 'hormat ka sorangan'
        elif tingkat == '2':
            label = 'hormat ka batur'
        return label


@app.route('/')  # Menampilkan menu klasifikasi
def home():
    return render_template('index.html')


@app.route('/tentang')  # Menampilkan menu tentang aplikasi
def tentang():
    return render_template('about.html')


# proses klasfifikasi dan koreksi
@app.route('/prosescek', methods=['POST', 'GET'])
def prosescek():
    if request.method == 'POST':
        label = ''
        text = request.form['text']
        tingkat = request.form['input']
        label = CekTingkat(tingkat)
        md = Modelnv(text, tingkat)
        preprocess = md.Preprocessing()
        token = preprocess
        pre_token = dict.fromkeys(token, "")

        process = md.Predict(preprocess)

        correct = md.Correct(tingkat, label, process)

        pre_token.update(correct)
        return json.dumps(pre_token)


if __name__ == '__main__':
    app.run(debug=True)
