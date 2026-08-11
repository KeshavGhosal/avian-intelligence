"""
Google Gemini API Integration Service for Bird Ecology and Habitat Information.

This module uses the official Google Gen AI Python SDK (`google-genai`) to query active Gemini models
(e.g., `gemini-3.6-flash` or `gemini-1.5-flash`) using strict system prompting and structured Pydantic schemas 
to return validated habitat, geographic range, migratory patterns, and biological fun facts.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger("bird_classifier.services")


class EcologyResponse(BaseModel):
    """Structured Pydantic schema for species ecology response."""
    scientific_name: str = Field(..., description="Scientific name of the bird species")
    common_name: str = Field(..., description="Common name of the bird species")
    habitat: str = Field(..., description="Description of natural habitat and environment")
    range: str = Field(..., description="Geographic range and global distribution")
    migratory_pattern: str = Field(..., description="Migratory habits and seasonal patterns")
    fun_fact: str = Field(..., description="Interesting biological or behavioral fact")


SYSTEM_PROMPT = """
You are an expert ornithologist and ecological data assistant.
Your task is to provide comprehensive, accurate ecological information for a given bird species.
Ensure all fields are strictly filled with detailed, factual data.
"""


class EcologyService:
    """
    Service layer interacting with Google Gemini API to fetch structured ecological data.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = (
            api_key 
            or os.getenv("GEMINI_API_KEY") 
            or os.getenv("GOOGLE_API_KEY") 
        )
        self.model_name = model_name
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY is not set. Ecology requests will fail or use fallback.")

    async def fetch_ecology_info(self, species_query: str) -> Dict[str, Any]:
        """
        Queries Gemini for detailed species ecological data and returns structured Pydantic output.
        
        Args:
            species_query: Name of the bird species (scientific or common name).
            
        Returns:
            Dict matching EcologyResponse schema fields.
        """
        if not self.client:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in environment variables."
            )

        user_message = f'Provide comprehensive ecology and habitat details for the bird species: "{species_query}".'

        try:
            logger.info(f"Querying Gemini model '{self.model_name}' for species: '{species_query}'")
            
            # Execute structured API call via async client
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=EcologyResponse,
                )
            )

            # Native parsing via Pydantic model response
            if hasattr(response, "parsed") and response.parsed:
                if isinstance(response.parsed, EcologyResponse):
                    return response.parsed.model_dump()
                elif isinstance(response.parsed, dict):
                    return EcologyResponse(**response.parsed).model_dump()

            # Fallback text parsing if response.parsed is not populated
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            parsed_data = json.loads(raw_text)
            validated_response = EcologyResponse(**parsed_data)
            return validated_response.model_dump()

        except json.JSONDecodeError as err:
            logger.error(f"Failed to parse JSON response from Gemini: {err}")
            raise RuntimeError("Model returned invalid JSON format for ecology data.")
        except Exception as err:
            logger.error(f"Error in EcologyService Gemini retrieval: {str(err)}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve ecology information: {str(err)}")