"""
Run the FastAPI backend with hot reload.

Loads `.env` from the project root so Azure OpenAI and other secrets
are available without exporting them in the shell.
"""
from dotenv import load_dotenv
import uvicorn

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
