# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

# Initialize app and load model
app = FastAPI(title="House Price Web App")
model = joblib.load("model/house_price_model.pkl")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Mount static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Redirect GET requests to /predict back to home
@app.get("/predict")
def redirect_predict():
    return RedirectResponse("/")

# Handle POST from form submission
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):
    X = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(X)[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": round(float(prediction), 3)
    })
