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


def get_fallback_ecology(species_query: str) -> Dict[str, Any]:
    """Generates structured fallback ecological data when API authentication is unavailable."""
    clean_name = species_query.strip().title()
    return {
        "scientific_name": clean_name,
        "common_name": clean_name,
        "habitat": f"Native to diverse natural ecosystems, dense woodlands, and coastal wetlands typical for {clean_name}.",
        "range": f"Distributed across regional ecosystems and migratory corridors where {clean_name} thrives.",
        "migratory_pattern": "Exhibits seasonal migratory behavior, traveling between summer breeding grounds and warmer wintering zones.",
        "fun_fact": f"The {clean_name} is known for its distinct flight patterns, territorial vocalizations, and specialized foraging adaptations."
    }


class EcologyService:
    """
    Service layer interacting with Google Gemini API to fetch structured ecological data.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (
            api_key 
            or os.getenv("GEMINI_API_KEY") 
            or os.getenv("GOOGLE_API_KEY") 
        )
        self.model_name = (
            model_name 
            or os.getenv("GEMINI_MODEL") 
            or "gemini-2.5-flash"
        )
        
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize GenAI client: {e}")
                self.client = None
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY is missing or unconfigured. Fallback ecological mode enabled.")

    async def fetch_ecology_info(self, species_query: str) -> Dict[str, Any]:
        """
        Queries Gemini for detailed species ecological data and returns structured Pydantic output.
        Falls back seamlessly if API key is invalid or unauthenticated.
        """
        if not self.client:
            logger.info(f"Using fallback ecological data for '{species_query}' (API key not configured).")
            return get_fallback_ecology(species_query)

        user_message = f'Provide comprehensive ecology and habitat details for the bird species: "{species_query}".'

        try:
            logger.info(f"Querying Gemini model '{self.model_name}' for species: '{species_query}'")
            
            # Try primary model attempt
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

        except Exception as err:
            logger.warning(f"Gemini API call failed ({err}). Switching to graceful ecological fallback for '{species_query}'.")
            return get_fallback_ecology(species_query)


CHAT_SYSTEM_PROMPT = """
You are Avian, a friendly and knowledgeable bird expert assistant embedded in a bird identification web app. 
You speak in a warm, natural, conversational tone — like a knowledgeable friend who loves birds, not like a formal bot or encyclopedia.

Rules:
- Never return JSON, bullet points, or markdown formatting. Write plain, flowing sentences only.
- Keep responses concise — 2 to 4 sentences maximum unless the user asks for more detail.
- If someone asks a general question (like "do they migrate?" or "what do they eat?"), assume they are asking about the bird the app just identified, or birds in general if no context is given.
- If someone greets you or makes small talk, respond naturally and warmly.
- You may occasionally use light humor or express genuine enthusiasm for birds.
- Never say "As an AI" or "I am a language model". Just be Avian, the bird expert.
"""


def get_fallback_chat_response(message: str) -> str:
    """Returns a friendly fallback message when the Gemini API is unavailable."""
    msg = message.lower().strip()
    if any(w in msg for w in ["hi", "hello", "hey", "good"]):
        return "Hey there! I'm Avian, your bird expert! Ask me anything about the bird we just identified, or any other bird you're curious about."
    if any(w in msg for w in ["migrate", "migration", "travel"]):
        return "Most birds follow incredible migratory routes twice a year — some travel thousands of miles between their breeding and wintering grounds! It's one of nature's most remarkable feats."
    if any(w in msg for w in ["eat", "food", "diet", "feed"]):
        return "Bird diets vary wildly by species — from seeds and berries to insects, fish, and even other birds! The beak shape is usually a dead giveaway for what a bird eats."
    if any(w in msg for w in ["habitat", "live", "where", "home"]):
        return "Birds have adapted to virtually every habitat on Earth — dense rainforests, open oceans, deserts, and even Arctic tundra. Each species has evolved perfectly for its environment."
    if any(w in msg for w in ["sing", "call", "sound", "voice", "chirp"]):
        return "Birdsong is one of the most complex forms of animal communication! Many species learn their songs from their parents, and some can even mimic other birds or environmental sounds."
    return "That's a great question! Birds are endlessly fascinating creatures. Feel free to ask me anything specific about their behavior, habitat, migration, or diet and I'll do my best to help!"


class ChatService:
    """
    Conversational AI assistant service using Google Gemini for free-form bird-related questions.
    Returns plain, natural human-like text responses — not structured data.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or "gemini-2.5-flash"
        )

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"ChatService: Failed to initialize GenAI client: {e}")
                self.client = None
        else:
            self.client = None
            logger.warning("ChatService: GEMINI_API_KEY not configured. Fallback chat mode enabled.")

    async def chat(self, message: str, species_context: Optional[str] = None) -> str:
        """
        Sends a free-form user message to Gemini and returns a conversational plain-text response.
        Optionally accepts a species_context string (the bird species currently displayed in UI).
        Falls back gracefully if the API is unavailable.
        """
        if not self.client:
            logger.info(f"ChatService: Using fallback for message: '{message}'")
            return get_fallback_chat_response(message)

        context_prefix = f"[Currently identified bird: {species_context}]\n" if species_context else ""
        full_message = f"{context_prefix}User question: {message}"

        try:
            logger.info(f"ChatService: Querying Gemini '{self.model_name}' for: '{message}'")
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_message,
                config=types.GenerateContentConfig(
                    system_instruction=CHAT_SYSTEM_PROMPT,
                    temperature=0.8,
                )
            )
            reply = response.text.strip() if response.text else get_fallback_chat_response(message)
            return reply

        except Exception as err:
            logger.warning(f"ChatService: Gemini call failed ({err}). Using fallback.")
            return get_fallback_chat_response(message)
