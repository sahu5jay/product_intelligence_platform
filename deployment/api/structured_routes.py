from flask import Blueprint, request, render_template
import logging

from src.structured_ml.pipeline.prediction_pipeline import PredictPipeline
from src.structured_ml.pipeline.prediction_pipeline import CustomData

structured_bp = Blueprint("structured_bp", __name__)

@structured_bp.route("/predict", methods=["GET", "POST"])
def predict_route():

    if request.method == "GET":
        return render_template("predict.html")

    try:

        data = CustomData(
            GrLivArea=float(request.form.get('GrLivArea')),
            OverallQual=float(request.form.get('OverallQual')),
            YearBuilt=float(request.form.get('YearBuilt')),
            TotalBsmtSF=float(request.form.get('TotalBsmtSF')),
            GarageCars=float(request.form.get('GarageCars')),
            Neighborhood=request.form.get('Neighborhood'),
            ExterQual=request.form.get('ExterQual'),
            KitchenQual=request.form.get('KitchenQual')
        )

        final_new_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()

        logging.info("======----000>>>>>")

        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template("predict.html", final_result=results)

    except Exception as e:
        logging.error(e)
        return render_template("predict.html", final_result="Prediction Error")