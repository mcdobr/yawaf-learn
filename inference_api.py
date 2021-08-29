import logging
import sys

import joblib
from flask import Flask, request, json

from compare_models import custom_features, NORMAL_LABEL

model = joblib.load('models/decision_tree.joblib')
app = Flask(__name__)

@app.route("/predictions", methods=['POST'])
def predict():
    data = request.get_json()
    logging.info(data)

    # Cache - control, Cookie, Accept - Language, Pragma, Version, Connection, Scheme,
    # Content - Type, Accept, Accept - Encoding, Accept - Charset, User - Agent, Class

    raw_request = {
        "Method": data["method"],
        "Host": data["host"],
        "Path": data["path"],
        "Query": data["query"],
        "Body": data["body"],
    }

    for header in data["headers"]:
        raw_request[header["key"]] = header["value"]

    logging.info("Raw request: {}".format(raw_request))

    feature_vector = custom_features([raw_request])
    feature_vector = feature_vector.drop(columns=["Label"])
    logging.info('Feature vector: {}'.format(feature_vector))
    logging.debug('Model {}'.format(model))

    prediction =  'NORMAL' if model.predict(feature_vector)[0] == NORMAL_LABEL else 'ANOMALY'
    logging.info(prediction)

    return app.response_class(response=json.dumps({'label': prediction}), status=200, mimetype='application/json')


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(funcName)s:  %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("log/inference_api.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    app.run(host='0.0.0.0', port=11000, debug=True)
