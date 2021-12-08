from flask import Flask, render_template

import settings


app = Flask('MyPortfolio')
app.config.from_object(settings.Dev)

@app.route('/', methods=['GET', 'POST'])
def login():
    data = {"msg": "Welcome to my site!"}
    return render_template("index.html", **data)

if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True, threaded=True)

    # gunicorn --config gunicorn.conf main:app