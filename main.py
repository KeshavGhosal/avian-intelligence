"""
Bird Identification Assistant - FastAPI Backend Service.

This main entrypoint defines the FastAPI web application, configures CORS middleware,
initializes the PyTorch bird inference engine and Anthropic Ecology service,
and exposes RESTful API endpoints for species classification and ecology intelligence.
"""

import os
import io
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

from model import BirdInferenceEngine
from services import EcologyService, EcologyResponse, ChatService

# Explicitly load the primary .env file to prevent loading .env.example or cached system vars
load_dotenv(dotenv_path=".env", override=True)

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bird_classifier.main")

# Global service instances
inference_engine: BirdInferenceEngine = None
ecology_service: EcologyService = None
chat_service: ChatService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for initializing model and services on startup."""
    global inference_engine, ecology_service, chat_service
    logger.info("Initializing application resources...")
    
    # Target state dict weights file
    model_path = os.getenv("MODEL_PATH", "final_bird_weights.pth")
    inference_engine = BirdInferenceEngine(checkpoint_path=model_path)
    ecology_service = EcologyService()
    chat_service = ChatService()
    
    logger.info("Application startup completed successfully.")
    yield
    logger.info("Shutting down application resources.")

app = FastAPI(
    title="Bird Identification Assistant API",
    description="Production FastAPI backend for CUB-200 species classification & Claude ecology retrieval.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS Middleware
cors_origins_str = os.getenv("CORS_ORIGINS", "*")
origins = [origin.strip() for origin in cors_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Data Transfer Objects ---

class TaxonomySchema(BaseModel):
    order: str = Field(..., description="Taxonomic Order")
    family: str = Field(..., description="Taxonomic Family")
    genus: str = Field(..., description="Taxonomic Genus")


class ClassifyResponse(BaseModel):
    species: str = Field(..., description="Scientific name of the bird species")
    common_name: str = Field(..., description="Common name of the bird species")
    confidence: float = Field(..., description="Model prediction confidence score (0 to 1)")
    taxonomy: TaxonomySchema = Field(..., description="Taxonomic breakdown details")


class EcologyRequest(BaseModel):
    species: str = Field(..., min_length=1, description="Bird species name (common or scientific)")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's conversational message or question")
    species_context: str = Field(default="", description="Currently identified bird species for context")


# --- API Routes ---

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serves the frontend web interface."""
    return FileResponse("index.html")


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint to verify backend status."""
    return {"status": "online", "service": "Bird Identification Assistant API"}


@app.post(
    "/api/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    tags=["Classification"]
)
async def classify_bird(file: UploadFile = File(...)) -> ClassifyResponse:
    """
    Accepts an uploaded image file (multipart/form-data), processes it via PyTorch EVA-02 Base (448x448),
    and returns top prediction species, common name, confidence score, and taxonomy.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image format (e.g. JPEG, PNG, WEBP)."
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        image = Image.open(io.BytesIO(file_bytes))
        
        # Execute PyTorch model inference
        result = inference_engine.predict(image)
        return ClassifyResponse(**result)

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not identify or open the uploaded image file."
        )
    except Exception as e:
        logger.error(f"Error during classification endpoint execution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {str(e)}"
        )


@app.post(
    "/api/ecology",
    response_model=EcologyResponse,
    status_code=status.HTTP_200_OK,
    tags=["Ecology"]
)
async def get_species_ecology(request: EcologyRequest) -> EcologyResponse:
    """
    Accepts species name JSON payload, queries Anthropic Claude for habitat & range metadata,
    and returns validated structured JSON response.
    """
    try:
        data = await ecology_service.fetch_ecology_info(request.species)
        return EcologyResponse(**data)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(err)
        )
    except Exception as err:
        logger.error(f"Error during ecology endpoint execution: {str(err)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ecology retrieval failed: {str(err)}"
        )


@app.post(
    "/api/chat",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    tags=["Chat"]
)
async def chat_with_assistant(request: ChatRequest) -> dict:
    """
    Accepts a free-form user message and an optional species context string,
    queries the ChatService (Google Gemini) with a conversational system prompt,
    and returns a plain natural-language response as JSON.
    """
    try:
        reply = await chat_service.chat(
            message=request.message,
            species_context=request.species_context or None
        )
        return {"reply": reply}
    except Exception as err:
        logger.error(f"Chat endpoint error: {str(err)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(err)}"
        )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
