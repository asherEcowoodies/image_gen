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
    application.state.agent = None
    application.state.kie_service = None
    application.state.orchestrator = None
    application.state.startup_error = None

    try:
        settings = get_settings()
        agent = ContentGenerationAgent(settings=settings)
        kie_service = KieService(settings=settings)

        application.state.agent = agent
        application.state.kie_service = kie_service
        application.state.orchestrator = GenerationOrchestrator(
            agent=agent,
            kie_service=kie_service,
        )
    except Exception as exc:
        # Keep service bootable so health checks can pass and config issues are visible.
        application.state.startup_error = str(exc)

    try:
        yield
    finally:
        kie_service = getattr(application.state, "kie_service", None)
        agent = getattr(application.state, "agent", None)
        if kie_service is not None:
            await kie_service.close()
        if agent is not None:
            await agent.close()


app = FastAPI(title="Query to Image Agent API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    startup_error = getattr(app.state, "startup_error", None)
    if startup_error:
        return {"status": "degraded", "detail": startup_error}
    return {"status": "ok", "detail": "ready"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    startup_error = getattr(app.state, "startup_error", None)
    orchestrator: GenerationOrchestrator | None = getattr(app.state, "orchestrator", None)
    if startup_error or orchestrator is None:
        detail = startup_error or "Service is not initialized"
        raise HTTPException(status_code=503, detail=f"Service unavailable: {detail}")

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
