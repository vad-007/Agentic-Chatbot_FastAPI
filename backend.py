#Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
load_dotenv()

class RequestState(BaseModel):
    model_name :str
    model_provider :str
    system_prompt:str
    messages:List[str]
    allow_search:bool

#Step2: Setup AI Agent from FrontEnd Request
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODELS=["gpt-4o-mini","llama-3.3-70b-versatile"]

app = FastAPI(title="AI Agent API")

@app.post("/chat")
def chat_endpoint(request:RequestState):
        """
        API Endpoint to interact with the Chatbot using LangGraph and search tools.
        It dynamically selects the model specified in the request
        """
        if request.model_name not in ALLOWED_MODELS:
            return {"error":"Invalid Model Name. Please select from the following models: "+str(ALLOWED_MODELS)}
        
        llm_id = request.model_name
        query = request.messages
        allow_search = request.allow_search
        system_prompt = request.system_prompt
        provider = request.model_provider
        
        # Create AI Agent and get response from it!
        
        response=get_response_from_ai_agent(
            llm_id,query,allow_search,system_prompt,provider
        )
        return response

#Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)