from flask import Flask, render_template
from deployment.api.structured_routes import structured_bp
from deployment.api.gan_routes import gan_bp
from deployment.api.nlp_routes import nlp_bp

app = Flask(
    __name__,
    template_folder="../../frontend/templates",
    static_folder="../../frontend/static"
)

# -------------------
# Register API routes
# -------------------

app.register_blueprint(structured_bp)
app.register_blueprint(gan_bp)
app.register_blueprint(nlp_bp)


# -------------------
# Frontend Pages
# -------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/structured")
def structured_page():
    return render_template("predict.html")


@app.route("/generate")
def generate_page():
    """
    Render GAN generator page
    """
    return render_template("generate.html", labels=[], images=[])


@app.route("/nlp")
def nlp_page():
    return render_template("analyze.html")


# -------------------
# Run Server
# -------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)