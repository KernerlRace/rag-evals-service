"""Answer generation. Local by default (Ollama over HTTP); Amazon Bedrock when RAG_LLM_BACKEND=bedrock."""

from dataclasses import dataclass

import httpx

SYSTEM_PROMPT = (
    "You answer questions about Python using only the numbered documentation excerpts you are given. "
    "Answer in at most three sentences and cite the excerpts you used like [1] or [2]. "
    "If the excerpts do not contain the answer, reply exactly: I don't know."
)


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


def build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}\nAnswer:"


class OllamaLLM:
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT, json_mode: bool = False) -> Generation:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        r = self.client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        text = data["message"]["content"].strip()
        return Generation(text, data.get("prompt_eval_count", 0), data.get("eval_count", 0))


class BedrockLLM:
    """Bedrock's Converse API has the same request shape for every model, so the model is just config."""

    name = "bedrock"

    def __init__(self, model_id: str, region: str):
        import boto3

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT, json_mode: bool = False) -> Generation:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 512},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        usage = resp.get("usage", {})
        return Generation(text, usage.get("inputTokens", 0), usage.get("outputTokens", 0))


class ExtractiveLLM:
    """No model at all: returns the first sentence of the first passage. For tests and as a floor baseline."""

    name = "extractive"

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT, json_mode: bool = False) -> Generation:
        first = prompt.split("[1] ", 1)[-1].split("\n\n", 1)[0]
        body = first.split("\n", 1)[-1]  # drop the "title > section" header line
        sentence = body.split(". ", 1)[0].strip()
        return Generation(sentence + " [1]", len(prompt.split()), len(sentence.split()))


def make_llm(settings):
    if settings.llm_backend == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if settings.llm_backend == "bedrock":
        return BedrockLLM(settings.bedrock_model_id, settings.aws_region)
    if settings.llm_backend == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown RAG_LLM_BACKEND {settings.llm_backend!r}")
