from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.agent.content_agent import ContentGenerationAgent
from app.services.exceptions import AgentOutputError, KieServiceError, KieTimeoutError
from app.services.kie_service import KieService
from app.services.orchestrator import GenerationOrchestrator
from app.utils.config import get_settings
from app.utils.models import GenerateRequest, GenerateResponse


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    agent = ContentGenerationAgent(settings=settings)
    kie_service = KieService(settings=settings)

    application.state.agent = agent
    application.state.kie_service = kie_service
    application.state.orchestrator = GenerationOrchestrator(
        agent=agent,
        kie_service=kie_service,
    )

    try:
        yield
    finally:
        await kie_service.close()
        await agent.close()


app = FastAPI(title="Query to Image Agent API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    orchestrator: GenerationOrchestrator = app.state.orchestrator

    try:
        return await orchestrator.generate(query=payload.query)
    except AgentOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except KieTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except KieServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc
