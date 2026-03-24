import os
from typing import Any, Dict, Optional

from promptulate.llms._litellm import LiteLLM
from promptulate.llms.base import BaseLLM


class LLMFactory:
    @classmethod
    def build(
        cls, model_name: str, *, model_config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> BaseLLM:
        model_config = model_config or {}

        if "/" in model_name:
            provider, model_id = model_name.split("/", 1)
            provider = provider.lower()

            if provider == "zhipu":
                from promptulate.llms import ZhiPu

                return ZhiPu(model=model_id, model_config=model_config, **kwargs)
            elif provider == "qianfan":
                from promptulate.llms import QianFan

                return QianFan(model=model_id, model_config=model_config, **kwargs)
            elif provider == "forge":
                if "/" not in model_id:
                    raise ValueError(
                        "Forge model must use format `forge/Provider/model-name`."
                    )

                forge_model_config = {
                    "api_base": os.getenv(
                        "FORGE_API_BASE", "https://api.forge.tensorblock.co/v1"
                    ),
                    **model_config,
                }
                if (
                    "api_key" not in forge_model_config
                    and os.getenv("FORGE_API_KEY") is not None
                ):
                    forge_model_config["api_key"] = os.getenv("FORGE_API_KEY")

                return LiteLLM(
                    model=f"openai/{model_id}",
                    model_config=forge_model_config,
                    **kwargs,
                )

        return LiteLLM(model=model_name, model_config=model_config, **kwargs)
