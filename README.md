# CodexOpenAI

This repository contains an example weather agent built with the OpenAI Agent SDK.

## Weather Agent

Install dependencies and run the script, optionally passing a city name:

```bash
pip install openai requests
export OPENAI_API_KEY=your_api_key
python weather_agent.py London
```

If no city is provided, the agent queries the weather in Paris by default.
