from fastapi import FastAPI
from voiceflip.api.guardrail_router import router as guardrail_router
from voiceflip.api.state import HEALTH_STATS

app = FastAPI(
    title="Citation Guardrail Engine",
    description="A highly deterministic microservice for RAG post-processing.",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """
    Returns the service status and counters for each citation decision type.
    """
    return {"status": "healthy", "counters": HEALTH_STATS}


app.include_router(guardrail_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("voiceflip.api.main:app", host="0.0.0.0", port=8000, reload=True)
