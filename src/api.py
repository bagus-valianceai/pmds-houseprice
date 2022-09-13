from flask import Flask

app = Flask(__name__)

@app.route('/')
def welcome():
    return "<h1 style='color:blue'>Hello There!</h1>"

def main():
    app.run(host = '0.0.0.0', port = 8080)

if __name__ == "__main__":
    main()