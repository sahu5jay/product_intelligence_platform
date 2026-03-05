from flask import Flask, request, render_template, jsonify, redirect, url_for
from src.structured_ml.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def home_page():
    return redirect(url_for('predict_datapoint'))


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('form.html')

    else:
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
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('form.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)