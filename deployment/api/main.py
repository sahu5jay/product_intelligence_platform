from flask import Flask, render_template, send_from_directory
from deployment.api.structured_routes import structured_bp
# from deployment.api.gan_routes import gan_bp
# from deployment.api.nlp_routes import nlp_bp

app = Flask(
    __name__,
    template_folder="../../frontend/templates",
    static_folder="../../frontend/static"
)

# Register API routes
app.register_blueprint(structured_bp)
# app.register_blueprint(gan_bp)
# app.register_blueprint(nlp_bp)


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
    return render_template("generate.html")


@app.route("/nlp")
def nlp_page():
    return render_template("analyze.html")


# -------------------
# Static Files
# -------------------

# @app.route("/static/<path:filename>")
# def static_files(filename):
#     return send_from_directory("../../frontend/static", filename)


# -------------------
# Run Server
# -------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)