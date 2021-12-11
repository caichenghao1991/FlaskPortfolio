from flask import Flask, render_template
from flask import request
import settings


app = Flask('MyPortfolio')
app.config.from_object(settings.Dev)

@app.route('/', methods=['GET', 'POST'])
def login():
    data = {"msg": "Welcome to my site!"}
    return render_template("index.html", **data)


@app.route('/project1', methods=['GET', 'POST'])
def project1():
    #stock = request.args.get('magic', 'no')
    stock = ''
    if request.method == 'POST':
        stock = request.form.get('stock', '')
        print(stock)
        return render_template('project1.html', **locals())

    return render_template('project1.html', **locals())

if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True, threaded=True)

    # gunicorn -c config.py main:app