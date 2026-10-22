
from pydantic import BaseModel, Field


class JobPredictionRequest(BaseModel):
    job_text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Complete job posting text",
    )
    telecommuting: int = Field(default=0, ge=0, le=1)
    has_company_logo: int = Field(default=0, ge=0, le=1)
    has_questions: int = Field(default=0, ge=0, le=1)


class JobPredictionResponse(BaseModel):
    prediction: str
    fraud_score: float
    fraud_score_percent: float
    warning: str
