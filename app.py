from flask import Flask, jsonify, request, render_template
# import flask_socketio
from flask_socketio import SocketIO
from model_predict import load_iamge_for_model, predict
import json


app = Flask(__name__, static_url_path='/static')  # Construct an instance of Flask class for our webapp
app.config['SECRET_KEY'] = 'SECRET'
socketio = SocketIO(app,logger=True, engineio_logger=True)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/model", methods=['GET', 'POST'])
def model():
	return render_template('model.html')

@socketio.on('connect_event')
def socket_connect(data):
	print("Socket Connected!")


@socketio.on('file_transfer')
def file_transfer(data):
	img = load_iamge_for_model(data['img'])
	prediction = predict(img)
	ret_data = json.dumps({'prediction' : str(prediction), 'amount_of_pass' : str(data['amount_of_pass'])})
	try:
		socketio.emit("prediction",ret_data)
	except:
		pass
    

if __name__ == '__main__':  
   socketio.run(app, debug=True, host="localhost")

