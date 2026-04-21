from __future__ import annotations

from app.agent.content_agent import ContentGenerationAgent
from app.services.kie_service import KieService
from app.utils.models import GenerateResponse


class GenerationOrchestrator:
    def __init__(self, agent: ContentGenerationAgent, kie_service: KieService) -> None:
        self._agent = agent
        self._kie_service = kie_service

    async def generate(self, query: str) -> GenerateResponse:
        agent_result = await self._agent.generate(query=query)
        image_url = await self._kie_service.generate_image(image_prompt=agent_result.image_prompt)

        return GenerateResponse(
            image_prompt=agent_result.image_prompt,
            captions=agent_result.captions,
            hashtags=agent_result.hashtags,
            image_url=image_url,
        )
