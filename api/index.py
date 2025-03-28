from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import json
import logging
from typing import Dict, Optional, List, Any, Union
import requests
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
import re

# Cargar variables de entorno desde el directorio raíz
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
GEMINI_API_KEYS = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY2")]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    chat_id: Optional[str] = None
    query: str
    options: Optional[Dict] = None

class ConversationResponse(BaseModel):
    chat_id: str
    result: str
    conversation_history: List[Message]

# Almacén de conversaciones
conversations = {}

class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = [key for key in api_keys if key]
        if not self.api_keys:
            raise ValueError("No API keys found in environment variables")
        self.current_key_index = 0
        
    def get_current_key(self):
        return self.api_keys[self.current_key_index]
    
    def switch_to_next_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return self.get_current_key()

class QuotaExceededException(Exception):
    pass

class SearchEngine:
    def __init__(self, api_key, search_engine_id):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query, num_results=5):
        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": num_results
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            logger.info(f"Search query: '{query}' - Results found: {'items' in search_results}")
            
            if "items" not in search_results:
                return []
                
            results = []
            for item in search_results["items"]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "displayLink": item.get("displayLink", "")
                }
                
                if "pagemap" in item and "cse_thumbnail" in item["pagemap"] and len(item["pagemap"]["cse_thumbnail"]) > 0:
                    result["thumbnail"] = item["pagemap"]["cse_thumbnail"][0].get("src", "")
                
                results.append(result)
                
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching Google: {str(e)}")
            return []

class AIHandler:
    def __init__(self, api_manager, search_engine):
        self.api_manager = api_manager
        self.search_engine = search_engine
        self.configure_model()

    def configure_model(self):
        self.api_key = self.api_manager.get_current_key()
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel('gemini-2.0-flash',
            generation_config={
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 8192,
            })
        
        self.chat_model = genai.GenerativeModel('gemini-2.0-flash',
            generation_config={
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 8192,
            })

    async def generate_with_fallback(self, prompt, max_retries=3):
        retries = 0
        while retries < max_retries:
            try:
                response = self.model.generate_content(prompt)
                content = response.text.strip()
                if content:
                    return content
                else:
                    raise Exception("Empty response from LLM")
            except Exception as e:
                logger.error(f"Error in LLM call (attempt {retries+1}/{max_retries}): {str(e)}")
                
                if "quota exceeded" in str(e).lower() and retries < max_retries - 1:
                    self.api_manager.switch_to_next_key()
                    self.configure_model()
                    logger.info(f"Switched to next API key due to quota limit")
                    retries += 1
                elif retries < max_retries - 1:
                    retries += 1
                    backoff_time = 2 ** (retries - 1)
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    raise e
        raise QuotaExceededException("All API keys have exceeded their quota or max retries reached")

    async def chat_with_history(self, messages, max_retries=3):
        retries = 0
        while retries < max_retries:
            try:
                chat = self.chat_model.start_chat(history=[
                    {"role": msg.role, "parts": [msg.content]} for msg in messages
                ])
                
                # Obtener la última pregunta del usuario
                last_user_message = next((msg.content for msg in reversed(messages) if msg.role == "user"), None)
                
                if not last_user_message:
                    raise ValueError("No user message found in conversation history")
                
                response = chat.send_message(last_user_message)
                content = response.text.strip()
                
                if content:
                    return content
                else:
                    raise Exception("Empty response from LLM")
            except Exception as e:
                logger.error(f"Error in chat LLM call (attempt {retries+1}/{max_retries}): {str(e)}")
                
                if "quota exceeded" in str(e).lower() and retries < max_retries - 1:
                    self.api_manager.switch_to_next_key()
                    self.configure_model()
                    logger.info(f"Switched to next API key due to quota limit")
                    retries += 1
                elif retries < max_retries - 1:
                    retries += 1
                    backoff_time = 2 ** (retries - 1)
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    raise e
        raise QuotaExceededException("All API keys have exceeded their quota or max retries reached")

    async def generate_search_queries(self, query, conversation):
        """Genera mejores consultas de búsqueda basadas en la consulta original y conversación."""
        
        # Si la pregunta original es muy corta (1-3 palabras), usarla directamente
        if len(query.split()) <= 3:
            return [query]
            
        # Para preguntas más largas, usar IA para generar consultas más optimizadas para búsqueda
        prompt = f"""
        Tu tarea es convertir la siguiente pregunta en consultas de búsqueda web eficaces (3-5 palabras).
        
        Pregunta: "{query}"
        
        Crea 3 consultas diferentes que:
        1. Extraigan las palabras clave esenciales
        2. Sean frases que la gente usaría en Google
        3. Incluyan términos específicos (nombres, fechas, lugares) si aparecen
        4. Sean búsquedas que un humano haría para encontrar esa información
        
        Formato: Solo las consultas, una por línea
        """
        
        try:
            response = await self.generate_with_fallback(prompt)
            
            # Extraer consultas (una por línea)
            queries = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            # Siempre incluir la consulta original
            if query not in queries:
                queries.append(query)
                
            return queries
            
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            return [query]  # En caso de error, usar la consulta original

    async def is_recommendation_query(self, query):
        """Determina si la consulta es una solicitud de recomendación."""
        recommendation_patterns = [
            r'^recomienda[rme]*\s+\d+',
            r'^enumera[rme]*\s+\d+',
            r'^lista[rme]*\s+\d+',
            r'^dime\s+\d+',
            r'^cuéntame\s+\d+',
            r'mejores\s+\d+',
            r'top\s+\d+'
        ]
        
        for pattern in recommendation_patterns:
            if re.search(pattern, query.lower()):
                return True
                
        return False

    async def execute_smart_search(self, query, conversation):
        """Ejecuta una búsqueda inteligente con múltiples estrategias."""
        
        # Verificar si es una solicitud de recomendación que Gemini podría manejar mejor
        is_recommendation = await self.is_recommendation_query(query)
        if is_recommendation:
            logger.info(f"Query identified as recommendation request: {query}")
            # Para recomendaciones, intentar primero con búsqueda web
            # Si no hay buenos resultados, caeremos en Gemini
        
        # Generar consultas de búsqueda optimizadas
        search_queries = await self.generate_search_queries(query, conversation)
        logger.info(f"Generated search queries: {search_queries}")
        
        # Búsqueda secuencial con las consultas generadas
        all_results = []
        seen_links = set()
        
        for search_query in search_queries:
            # Evitar búsquedas duplicadas
            if all_results and search_query == query:
                continue
                
            logger.info(f"Executing search: '{search_query}'")
            results = await self.search_engine.search(search_query)
            
            # Agregar solo resultados únicos
            for result in results:
                if result['link'] not in seen_links:
                    all_results.append(result)
                    seen_links.add(result['link'])
            
            # Si ya encontramos suficientes resultados, parar
            if len(all_results) >= 8:
                break
        
        # Ordenar por relevancia
        # Asumimos que los primeros resultados de cada búsqueda son más relevantes
        return all_results[:10]

    async def process_request(self, query, chat_id=None):
        # Recuperar historial de la conversación o crear uno nuevo
        if chat_id and chat_id in conversations:
            conversation = conversations[chat_id]
        else:
            chat_id = str(uuid.uuid4())
            conversation = []
            conversations[chat_id] = conversation
        
        # Ejecutar búsqueda inteligente
        search_results = await self.execute_smart_search(query, conversation)
        logger.info(f"Found {len(search_results)} search results")
        
        # Si no hay resultados o son insuficientes, usar Gemini directamente
        if len(search_results) < 2:
            # Verificar si la consulta es una solicitud de recomendación
            is_recommendation = await self.is_recommendation_query(query)
            
            # Crear prompt basado en el tipo de consulta
            if is_recommendation:
                prompt = f"""
                Eres un asistente de IA que proporciona recomendaciones personalizadas.
                
                El usuario ha solicitado: "{query}"
                
                Proporciona una respuesta útil y específica. Si la solicitud pide un número específico de recomendaciones, proporciona exactamente ese número.
                
                Formato: Comienza con una breve introducción y luego proporciona las recomendaciones numeradas.
                """
            else:
                prompt = f"""
                Responde a esta pregunta del usuario de manera conversacional:
                
                "{query}"
                
                Usa tu conocimiento general para proporcionar una respuesta útil, basada en hechos y bien explicada. Si no conoces la respuesta, o si la pregunta se refiere a eventos después de octubre de 2024, indícalo claramente.
                
                Estamos en marzo de 2025, así que puedes hablar con seguridad de eventos hasta esta fecha.
                """
            
            # Añadir contexto de la conversación si hay historial previo
            if conversation:
                context = "\nContexto de la conversación anterior:\n"
                for msg in conversation[-6:]:  # Últimos 6 mensajes
                    role = "Usuario" if msg.role == "user" else "Asistente"
                    context += f"{role}: {msg.content}\n"
                prompt = context + prompt
            
            try:
                response = await self.generate_with_fallback(prompt)
                
                # Actualizar historial
                conversation.append(Message(role="user", content=query))
                conversation.append(Message(role="assistant", content=response))
                
                return {
                    "chat_id": chat_id,
                    "result": response,
                    "conversation": conversation
                }
            except Exception as e:
                logger.error(f"Error using direct Gemini response: {str(e)}")
                # Si falla, continuar con búsqueda web si hay algún resultado
                if not search_results:
                    return {
                        "chat_id": chat_id,
                        "result": "Lo siento, no puedo responder a esa pregunta en este momento.",
                        "conversation": conversation
                    }
        
        # Procesar resultados de búsqueda
        search_context = ""
        for i, result in enumerate(search_results, 1):
            search_context += f"Resultado {i}:\n"
            search_context += f"Título: {result['title']}\n"
            search_context += f"URL: {result['link']}\n"
            search_context += f"Descripción: {result['snippet']}\n\n"
        
        # Extraer contexto de la conversación si existe
        conversation_context = ""
        if conversation:
            conversation_context = "Contexto reciente de la conversación:\n"
            for msg in conversation[-6:]:  # Últimos 6 mensajes
                role = "Usuario" if msg.role == "user" else "Asistente"
                conversation_context += f"{role}: {msg.content}\n"
        
        # Crear prompt para generar la respuesta
        prompt = f"""
        Eres un asistente conversacional avanzado. Estamos en marzo de 2025.
        
        {conversation_context if conversation_context else ""}
        
        La pregunta actual del usuario es: "{query}"
        
        Basándote en la siguiente información de búsqueda:
        
        {search_context}
        
        Genera una respuesta que:
        1. Sea correcta y basada en la información proporcionada
        2. Sea conversacional y natural, como si estuvieras chateando
        3. Sea completa y detallada, pero concisa
        4. NO mencione que estás basando tu respuesta en resultados de búsqueda
        5. Si los resultados de búsqueda no contienen la información necesaria para responder, dilo honestamente
        6. Si la pregunta pide recomendaciones y hay resultados insuficientes, proporciona algunas sugerencias generales
        
        Tu respuesta:
        """
        
        try:
            response = await self.generate_with_fallback(prompt)
            
            # Actualizar historial de conversación
            conversation.append(Message(role="user", content=query))
            conversation.append(Message(role="assistant", content=response))
            
            return {
                "chat_id": chat_id,
                "result": response,
                "conversation": conversation
            }
            
        except QuotaExceededException:
            fallback_response = "Lo siento, no puedo procesar tu solicitud en este momento."
            
            conversation.append(Message(role="user", content=query))
            conversation.append(Message(role="assistant", content=fallback_response))
            
            return {
                "chat_id": chat_id,
                "result": fallback_response,
                "conversation": conversation
            }

api_manager = APIKeyManager(GEMINI_API_KEYS)
search_engine = SearchEngine(GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID)
ai_handler = AIHandler(api_manager, search_engine)

@app.post("/chat", response_model=ConversationResponse)
async def chat(data: ConversationRequest):
    try:
        query = data.query.strip()
        logger.info(f"Received request: {query}, chat_id: {data.chat_id}")
        
        start_time = time.time()
        result = await ai_handler.process_request(query, data.chat_id)
        process_time = time.time() - start_time
        
        logger.info(f"Processing completed in {process_time:.2f} seconds")
        
        return {
            "chat_id": result["chat_id"],
            "result": result["result"],
            "conversation_history": result["conversation"]
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error procesando la solicitud. Por favor, intenta de nuevo.")

class CLIChat:
    def __init__(self, ai_handler):
        self.ai_handler = ai_handler
        self.chat_id = None
    
    async def start(self):
        print("¡Bienvenido al Chat CLI! Escribe 'salir' para terminar.")
        print("------------------------------------------------------")
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("\n¡Hasta luego!")
                    break
                
                print("\nProcesando...")
                
                result = await self.ai_handler.process_request(user_input, self.chat_id)
                self.chat_id = result["chat_id"]  # Mantener el ID de chat para la conversación
                
                print(f"\n{result['result']}")
                
            except Exception as e:
                print(f"\nError: {str(e)}")

async def cli_main():
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Modo interactivo
        cli_chat = CLIChat(ai_handler)
        await cli_chat.start()
    elif len(sys.argv) > 1:
        # Modo de consulta única
        query = " ".join(sys.argv[1:])
        print(f"Buscando: {query}")
        print("Procesando...")
        
        try:
            result = await ai_handler.process_request(query)
            print("\nResultado:")
            print(result["result"])
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Uso: python index.py 'tu consulta aquí'")
        print("   o: python index.py --interactive (para modo chat)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(cli_main())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)