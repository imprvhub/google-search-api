from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import logging
from typing import Dict, Optional, List
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import uuid

root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path)

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

class GeminiHandler:
    def __init__(self, api_manager):
        self.api_manager = api_manager
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

    async def process_request(self, query, chat_id=None):
        if chat_id and chat_id in conversations:
            conversation = conversations[chat_id]
        else:
            chat_id = str(uuid.uuid4())
            conversation = []
            conversations[chat_id] = conversation
        
        prompt = f"""
        Eres un asistente de IA conversacional. Responde a esta pregunta del usuario:
        
        "{query}"
        
        Usa tu conocimiento general para proporcionar una respuesta útil, basada en hechos y bien explicada. 
        Si no conoces la respuesta, o si la pregunta se refiere a eventos después de octubre de 2024, 
        indícalo claramente diciendo que tu conocimiento tiene un límite temporal.
        
        Estamos en marzo de 2025, así que menciona este detalle si es relevante para la pregunta.
        """
        
        if conversation:
            context = "\nContexto de la conversación anterior:\n"
            for msg in conversation[-6:]:
                role = "Usuario" if msg.role == "user" else "Asistente"
                context += f"{role}: {msg.content}\n"
            prompt = context + prompt
        
        try:
            response = await self.generate_with_fallback(prompt)
            
            conversation.append(Message(role="user", content=query))
            conversation.append(Message(role="assistant", content=response))
            
            return {
                "chat_id": chat_id,
                "result": response,
                "conversation": conversation
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            fallback_response = "Lo siento, no puedo procesar tu solicitud en este momento."
            
            conversation.append(Message(role="user", content=query))
            conversation.append(Message(role="assistant", content=fallback_response))
            
            return {
                "chat_id": chat_id,
                "result": fallback_response,
                "conversation": conversation
            }

api_manager = APIKeyManager(GEMINI_API_KEYS)
gemini_handler = GeminiHandler(api_manager)

@app.post("/chat", response_model=ConversationResponse)
async def chat(data: ConversationRequest):
    try:
        query = data.query.strip()
        logger.info(f"Received request: {query}, chat_id: {data.chat_id}")
        
        start_time = time.time()
        result = await gemini_handler.process_request(query, data.chat_id)
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
    def __init__(self, gemini_handler):
        self.gemini_handler = gemini_handler
        self.chat_id = None
    
    async def start(self):
        print("¡Bienvenido al Chat CLI de Gemini! Escribe 'salir' para terminar.")
        print("----------------------------------------------------------")
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("\n¡Hasta luego!")
                    break
                
                print("\nProcesando...")
                
                result = await self.gemini_handler.process_request(user_input, self.chat_id)
                self.chat_id = result["chat_id"]
                
                print(f"\n{result['result']}")
                
            except Exception as e:
                print(f"\nError: {str(e)}")

async def cli_main():
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        cli_chat = CLIChat(gemini_handler)
        await cli_chat.start()
    elif len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Consultando: {query}")
        print("Procesando...")
        
        try:
            result = await gemini_handler.process_request(query)
            print("\nResultado:")
            print(result["result"])
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Uso: python gemini_only_chatbot.py 'tu consulta aquí'")
        print("   o: python gemini_only_chatbot.py --interactive (para modo chat)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(cli_main())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)