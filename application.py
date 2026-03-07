from flask import Flask, request, render_template

# Structured ML
from src.structured_ml.pipeline.prediction_pipeline import CustomData
from src.structured_ml.pipeline.prediction_pipeline import PredictPipeline as StructuredPredictPipeline

# NLP
from src.nlp_module.pipeline.prediction_pipeline import PredictPipeline

# GAN
from src.gan_module.components.evaluation import GANEvaluator

from src.shared_utils.logger import logging


MODEL_PATH = "artifacts/gan/models/generator.pth"


application = Flask(__name__,
    template_folder="frontend/templates",
    static_folder="frontend/static")
app = application


# --------------------------------
# Home Page
# --------------------------------
@app.route('/')
def home_page():
    return render_template('home.html')


# --------------------------------
# Structured ML Prediction
# --------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('form.html')

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

        predict_pipeline = StructuredPredictPipeline()

        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('form.html', final_result=results)

    except Exception as e:
        logging.error(e)
        return render_template('form.html', final_result="Prediction Error")


# --------------------------------
# NLP Sentiment Prediction
# --------------------------------
@app.route('/text', methods=['GET', 'POST'])
def predict_sentiment():

    if request.method == 'GET':
        return render_template('text.html')

    try:

        review_text = request.form.get("review")
        print(f"-----==>>>{review_text}")

        predict_pipeline = PredictPipeline()

        prediction = predict_pipeline.predict(review_text)

        print(f"-----==>>>{prediction}")



        return render_template(
            'text.html',
            prediction_result=prediction
        )

    except Exception as e:
        logging.error(e)

        return render_template(
            'text.html',
            prediction_result="Prediction Error"
        )

@app.route("/generate", methods=["GET", "POST"])
def generate():

    images = []

    if request.method == "POST":

        num_images = int(request.form.get("num_images", 5))

        evaluator = GANEvaluator(MODEL_PATH)

        images = evaluator.generate_images(num_images)

    return render_template("generate.html", images=images)



# --------------------------------
# Run App
# --------------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)