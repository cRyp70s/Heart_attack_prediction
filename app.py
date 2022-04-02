from crypt import methods
from flask import Flask, jsonify, request
from heart_attack import train_classifier, predict

data_path = "data/heart_failure_clinical_records_dataset.csv"

classifer = train_classifier(data_path)

app = Flask(__name__)

@app.route("/predict-heart-attack", methods=["POST"])
def predictor():
    try:
        data = request.get_json(force=True)
        result = predict(classifer, data)
        return jsonify(result)
    except Exception as e:
        print(e)
        return {"msg": """Please provide the following fields in this order
                    [[
                     'age','anaemia','creatinine_phosphokinase',
                     'diabetes','ejection_fraction', 'high_blood_pressure',
                     'platelets','serum_creatinine','serum_sodium','sex',
                     'smoking','time'
                    ]...]
               """}, 400

if __name__ == "__main__":
    app.run()