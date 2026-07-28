import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

import datasets
from dotenv import load_dotenv

from client.providers import Query
from client.query import get_responses_from_models, ERROR_RESPONSE

RESPONSES_DIR = Path("../responses")
DATA_DIR = Path("data")

INVALID_ANS = "[invalid]"


def extract_json_list(text: str, expected_length: int) -> list[str] | str:
    match = re.search(r'(\[.*\])', text, flags=re.DOTALL)
    if match:
        match_str = match.group(0).strip()
        try:
            json_list = json.loads(match_str)
            if isinstance(json_list, list) and len(json_list) == expected_length:
                return json_list
        except:
            pass
        return INVALID_ANS


def generate_hard_negatives(num_samples: int, split: str) -> list[tuple[int, list[str]]]:
    dataset = datasets.load_dataset("wmt/wmt19", "zh-en", streaming=False)[split]
    pairs = [dataset[i]['translation'] for i in range(num_samples)]

    os.makedirs(RESPONSES_DIR, exist_ok=True)
    response_file = RESPONSES_DIR / Path(f"hard_negatives_{split}.json")
    queries = [Query(turns=[{"user": str(pair)}]) for pair in pairs]
    responses = asyncio.run(get_responses_from_models(
        queries,
        ["A"],
        response_file,
        workers_per_model=15,
        requests_per_minute=30,
    ))

    answers = [(i, extract_json_list(responses[i]["A"], expected_length=3)) for i in range(num_samples)]
    num_errors = sum(1 for i in range(num_samples) if responses[i]["A"] == ERROR_RESPONSE)
    valid_answers = [answer for answer in answers if answer[1] != INVALID_ANS]
    print(f"Results: {f'{num_errors}/{num_samples} api errors, ' if num_errors else ''}"
          f"{len(valid_answers)}/{num_samples - num_errors} non-error responses are valid "
          )

    return valid_answers


if __name__ == '__main__':
    load_dotenv()
    generate_hard_negatives(1000, split='validation')