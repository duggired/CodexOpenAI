"""Simple weather agent example using the OpenAI Agents SDK."""

from __future__ import annotations

import json
import requests
from openai import OpenAI


def get_current_weather(location: str) -> str:
    """Return current weather for a given location using wttr.in."""
    response = requests.get(f"https://wttr.in/{location}?format=j1", timeout=10)
    response.raise_for_status()
    data = response.json()
    condition = data["current_condition"][0]
    description = condition["weatherDesc"][0]["value"]
    temp_c = condition["temp_C"]
    return f"{description}, {temp_c}°C"


def main() -> None:
    client = OpenAI()

    agent = client.agents.create(
        name="Weather Agent",
        instructions="Use get_current_weather to answer questions about weather.",
        model="gpt-4o-mini",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Look up the current weather for a given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City to look up",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    )

    session = client.sessions.create(agent_id=agent.id)

    question = "What's the weather in Paris?"
    response = client.responses.create(
        agent_id=agent.id,
        session_id=session.id,
        input=question,
    )

    for item in response.output:
        if item.type == "tool_call":
            args = json.loads(item.input)
            result = get_current_weather(**args)
            response = client.responses.create(
                agent_id=agent.id,
                session_id=session.id,
                response_id=response.id,
                output=[
                    {
                        "role": "assistant",
                        "tool_call_id": item.id,
                        "type": "tool_result",
                        "content": [{"type": "output_text", "text": result}],
                    }
                ],
            )

    final_text = response.output[0].content[0].text
    print(final_text)


if __name__ == "__main__":
    main()
