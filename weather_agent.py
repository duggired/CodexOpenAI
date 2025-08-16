"""Simple weather agent example using the OpenAI Agents SDK."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

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


def run_agent(question: str) -> str:
    """Run the weather agent and return the model's response."""
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
    response = client.responses.create(
        agent_id=agent.id,
        session_id=session.id,
        input=question,
    )

    # Process tool calls until the model returns a final response
    while any(item.type == "tool_call" for item in response.output):
        new_output: list[dict[str, Any]] = []
        for item in response.output:
            if item.type == "tool_call":
                args = json.loads(item.input)
                result = get_current_weather(**args)
                new_output.append(
                    {
                        "role": "assistant",
                        "tool_call_id": item.id,
                        "type": "tool_result",
                        "content": [{"type": "output_text", "text": result}],
                    }
                )
        response = client.responses.create(
            agent_id=agent.id,
            session_id=session.id,
            response_id=response.id,
            output=new_output,
        )

    return response.output[0].content[0].text


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    parser = argparse.ArgumentParser(description="Ask the weather agent a question")
    parser.add_argument(
        "location",
        nargs="?",
        default="Paris",
        help="City to look up (default: Paris)",
    )
    args = parser.parse_args()

    question = f"What's the weather in {args.location}?"
    print(run_agent(question))


if __name__ == "__main__":
    main()
