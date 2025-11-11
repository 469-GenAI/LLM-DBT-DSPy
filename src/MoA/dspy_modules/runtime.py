"""
Runtime utilities for managing heterogeneous agent language models and personas.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional

import dspy


@dataclass
class AgentModelConfig:
    """Configuration describing how to instantiate a specific agent LM."""

    model_id: str
    model_name: str
    api_key: Optional[str] = None
    model_type: str = "chat"
    temperature: float = 1.0


@dataclass
class PersonaTemplate:
    """Persona template metadata used to specialise agent prompts."""

    persona: str
    system_preamble: str
    stylistic_guidance: str


class AgentLMRegistry:
    """Registry that lazily constructs and caches agent-specific language models."""

    def __init__(
        self,
        default_lm: dspy.LM,
        model_configs: Dict[str, AgentModelConfig],
    ):
        if default_lm is None:
            raise ValueError("Default language model is required.")

        self.default_lm = default_lm
        self.model_configs = model_configs
        self._cache: Dict[str, dspy.LM] = {}

    def get(self, model_id: Optional[str]) -> dspy.LM:
        """
        Retrieve the configured LM for a given model identifier.

        Args:
            model_id: Identifier produced by the planner. Falls back to default when unknown.
        """
        if model_id is None or model_id not in self.model_configs:
            return self.default_lm

        if model_id not in self._cache:
            config = self.model_configs[model_id]
            self._cache[model_id] = dspy.LM(
                config.model_name,
                model_type=config.model_type,
                api_key=config.api_key or getattr(self.default_lm, "api_key", None),
                temperature=config.temperature,
            )

        return self._cache[model_id]

    def describe_available_models(self) -> str:
        """Return a JSON string describing available model identifiers."""
        summary = {
            "default": {
                "model_name": getattr(self.default_lm, "model", None)
                or getattr(self.default_lm, "model_name", "unknown"),
                "model_type": getattr(self.default_lm, "model_type", "chat"),
            }
        }
        for model_id, config in self.model_configs.items():
            summary[model_id] = {
                "model_name": config.model_name,
                "model_type": config.model_type,
            }
        return json.dumps(summary, indent=2)


def parse_agent_model_configs(payload: Optional[str]) -> Dict[str, AgentModelConfig]:
    """
    Parse a JSON payload describing agent-specific model overrides.

    Expected format:
        {
            "agent_finance": {
                "model_name": "groq/llama-3.1-8b-instant",
                "api_key": "...",
                "model_type": "chat",
                "temperature": 0.9
            }
        }
    """
    if not payload:
        return {}

    try:
        raw_config = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid agent model configuration payload.") from error

    if not isinstance(raw_config, dict):
        raise ValueError("Agent model configuration must be a JSON object.")

    parsed: Dict[str, AgentModelConfig] = {}
    for model_id, config in raw_config.items():
        if not isinstance(config, dict):
            continue

        model_name = config.get("model_name")
        if not model_name:
            continue

        parsed[model_id] = AgentModelConfig(
            model_id=model_id,
            model_name=str(model_name),
            api_key=config.get("api_key"),
            model_type=str(config.get("model_type", "chat")),
            temperature=float(config.get("temperature", 1.0)),
        )

    return parsed


def parse_persona_templates(payload: Optional[str]) -> Dict[str, PersonaTemplate]:
    """
    Parse persona template configuration into PersonaTemplate objects.

    Example payload:
        {
            "narrator": {
                "system_preamble": "You are a captivating storyteller...",
                "stylistic_guidance": "Focus on emotional arcs and pacing."
            }
        }
    """
    if not payload:
        return {}

    try:
        raw_templates = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid persona template payload.") from error

    if not isinstance(raw_templates, dict):
        raise ValueError("Persona template payload must be a JSON object.")

    parsed: Dict[str, PersonaTemplate] = {}
    for persona, template in raw_templates.items():
        if not isinstance(template, dict):
            continue

        system_preamble = template.get("system_preamble")
        stylistic_guidance = template.get("stylistic_guidance")
        if not system_preamble and not stylistic_guidance:
            continue

        parsed[persona] = PersonaTemplate(
            persona=persona,
            system_preamble=str(system_preamble or ""),
            stylistic_guidance=str(stylistic_guidance or ""),
        )

    return parsed


def apply_persona_template(
    persona: Optional[str],
    system_message: str,
    user_prompt: str,
    templates: Dict[str, PersonaTemplate],
) -> Dict[str, str]:
    """
    Merge persona template guidance into the base system and user prompts.
    """
    if persona is None or persona not in templates:
        return {
            "system_message": system_message,
            "user_prompt": user_prompt,
        }

    template = templates[persona]
    persona_system = "\n\n".join(
        segment.strip()
        for segment in [template.system_preamble, system_message]
        if segment.strip()
    )
    persona_prompt = "\n\n".join(
        segment.strip()
        for segment in [user_prompt, template.stylistic_guidance]
        if segment.strip()
    )

    return {
        "system_message": persona_system,
        "user_prompt": persona_prompt,
    }

