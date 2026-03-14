import json
import httpx

from graph.debate.config import DebateConfig


class Agent:
    # thin wrapper around openrouter chat completions

    def __init__(self, model: str, system_prompt: str, config: DebateConfig):
        self.model = model
        self.system_prompt = system_prompt
        self.config = config
        self.history: list[dict] = []

    def _call(self, user_message: str, temperature: float) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        response = httpx.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000,
            },
            timeout=60,
        )
        if response.status_code != 200:
            print(f"api error {response.status_code}: {response.text}")
            response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # track conversation history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": content})

        return content

    def respond(self, message: str, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else 0.5
        return self._call(message, temp)


class Proposer(Agent):
    def __init__(self, config: DebateConfig):
        from graph.debate.prompts import PROPOSER_SYSTEM
        super().__init__(config.proposer_model, PROPOSER_SYSTEM, config)

    def propose(self, context: str) -> str:
        return self.respond(context, self.config.proposer_temperature)

    def rebut(self, challenge: str) -> str:
        from graph.debate.prompts import PROPOSER_REBUTTAL
        message = PROPOSER_REBUTTAL.format(challenge=challenge)
        return self.respond(message, self.config.proposer_temperature)


class Challenger(Agent):
    def __init__(self, config: DebateConfig):
        from graph.debate.prompts import CHALLENGER_SYSTEM
        super().__init__(config.challenger_model, CHALLENGER_SYSTEM, config)

    def challenge(self, context: str) -> str:
        return self.respond(context, self.config.challenger_temperature)

    def reassess(self, rebuttal: str) -> str:
        from graph.debate.prompts import CHALLENGER_REBUTTAL
        message = CHALLENGER_REBUTTAL.format(rebuttal=rebuttal)
        return self.respond(message, self.config.challenger_temperature)
