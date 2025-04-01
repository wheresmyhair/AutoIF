#!/usr/bin/env python3
import os
import json
import argparse
import random
import time
import glob
import re
from typing import Union, Tuple

from openai import OpenAI
from tqdm import tqdm
from functools import partial


def unified_api_call(
    prompt: str, 
    client: "OpenAI", 
    model: str, 
    max_tokens: int, 
    seed: int = 42,
    reasoning: bool = True
) -> Union[str|Tuple[str]]:
    """
    Makes an API call using the provided client and parameters.
    For clients with a 'chat' attribute (e.g., OpenAI, DeepSeek), it calls chat.completions.create in streaming mode.
    For Google, it uses models.generate_content.
    Based on mode:
      - "generate": returns cleaned generated text.
      - "reason": returns a tuple (answer, reasoning) by processing reasoning tokens correctly.
    """
    retry = 0
    while retry < 5:
        try:
            # Check if the client supports chat completions.
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                stream=True,
                seed=seed
            )
            if reasoning:
                answer_text = ""
                reasoning_text = ""
                for chunk in response:
                    # If a chunk has a reasoning token, add it to reasoning_text.
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        reasoning_text += chunk.choices[0].delta.reasoning_content
                    else:
                        content = chunk.choices[0].delta.content
                        if content:
                            answer_text += content
                return reasoning_text.strip(), answer_text.strip()
            
            else:
                full_response = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                return full_response.strip()
        
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            retry += 1
            time.sleep(10)
            
    return "[ERROR] API call failed after retries."


gen_client = OpenAI(
    api_key='sk-to2025llm',
    base_url='https://llm-gateway.tensoropera.ai/v1/',
    default_headers={
        'X-Chat-Mode': 'reasoning',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': 'llm-gateway.tensoropera.ai',
        'Connection': 'keep-alive'
    }
)
model_generate = 'DeepSeek-R1'

generate = partial(
    unified_api_call,
    client=gen_client,
    model=model_generate,
    max_tokens=None,
    reasoning=True
)

if __name__ == "__main__":
    reasoning, answer = generate(prompt="Which one is larger, 8.9 or 8.11?", seed=42)
    print(reasoning)
    print(answer)