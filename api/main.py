
from fastapi import FastAPI, HTTPException

from api.config import API_TITLE, API_DESCRIPTION, API_VERSION
from api.predictor import predict_job_fraud
from api.schemas import JobPredictionRequest, JobPredictionResponse


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


@app.get("/")
def home():
    return {
        "message": "Job Posting Fraud Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=JobPredictionResponse,
)
def predict_job(request: JobPredictionRequest):
    try:
        result = predict_job_fraud(
            job_text=request.job_text,
            telecommuting=request.telecommuting,
            has_company_logo=request.has_company_logo,
            has_questions=request.has_questions,
        )
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
