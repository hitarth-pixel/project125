from flask import Flask,jsonify,request
from project125 import get_prediction
app=Flask(__name__)
@app.route('/predict-application',methods=['POST'])
def predictFunction():
    image=request.files.get("alphabet")
    prediction=get_prediction(image)

    return jsonify({
        "prediction":prediction
    }),200

if __name__=="__main__":
    app.run(debug=True)