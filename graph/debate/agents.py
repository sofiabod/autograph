import time
import httpx

from graph.debate.config import DebateConfig

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubled each retry


class LLMAgent:
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

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
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
                    timeout=90,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    print(f"api returned {response.status_code}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                if response.status_code != 200:
                    print(f"api error {response.status_code}: {response.text}")
                    response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # track conversation history
                self.history.append({"role": "user", "content": user_message})
                self.history.append({"role": "assistant", "content": content})
                return content

            except httpx.TimeoutException as e:
                last_error = e
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"request timed out, retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"api call failed after {MAX_RETRIES} retries: {last_error}")

    def respond(self, message: str, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else 0.5
        return self._call(message, temp)


class Proposer(LLMAgent):
    def __init__(self, config: DebateConfig):
        from graph.debate.prompts import PROPOSER_SYSTEM
        super().__init__(config.proposer_model, PROPOSER_SYSTEM, config)

    def propose(self, context: str) -> str:
        return self.respond(context, self.config.proposer_temperature)

    def rebut(self, challenge: str) -> str:
        from graph.debate.prompts import PROPOSER_REBUTTAL
        message = PROPOSER_REBUTTAL.format(challenge=challenge)
        return self.respond(message, self.config.proposer_temperature)


class Challenger(LLMAgent):
    def __init__(self, config: DebateConfig):
        from graph.debate.prompts import CHALLENGER_SYSTEM
        super().__init__(config.challenger_model, CHALLENGER_SYSTEM, config)

    def challenge(self, context: str) -> str:
        return self.respond(context, self.config.challenger_temperature)

    def reassess(self, rebuttal: str) -> str:
        from graph.debate.prompts import CHALLENGER_REBUTTAL
        message = CHALLENGER_REBUTTAL.format(rebuttal=rebuttal)
        return self.respond(message, self.config.challenger_temperature)
