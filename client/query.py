import asyncio
import json
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path

from google.genai.types import GenerateContentResponse
from tqdm.asyncio import tqdm_asyncio

from client.providers import get_provider, Query

ERROR_RESPONSE = "[error]"


class RateLimiter:

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.lock = asyncio.Lock()
        self.times = deque(maxlen=requests_per_minute)

    async def enter(self):
        async with self.lock:
            current_time = time.perf_counter()
            if len(self.times) == self.rpm:
                elapsed_time = current_time - self.times[0]
                if elapsed_time >= 59:  # -1 for safety
                    await asyncio.sleep(elapsed_time - 59)
            self.times.append(time.perf_counter())


async def query_model(
        model_id: str,
        query: Query,
        rate_limiter: RateLimiter = None,
) -> GenerateContentResponse:
    # Get the appropriate provider
    provider_name, provider = get_provider(model_id=model_id)
    # Make the request to the LLM provider
    if rate_limiter:
        await rate_limiter.enter()
    response = await provider.query(model_id=model_id, query=query)
    return response


async def query_model_with_backoff(
        model_id: str,
        query: Query,
        max_attempts: int = 5,
        tqdm_pos: int = 1,
        rate_limiter: RateLimiter = None,
) -> GenerateContentResponse:
    """
    Attempts query max_attempts times before giving up, using exponential backoff.
    """
    delay = 2.0
    try:
        return await query_model(model_id=model_id, query=query)
    except Exception as e:
        with tqdm_asyncio(range(2, max_attempts + 1), total=max_attempts, initial=1, leave=False, position=tqdm_pos,
                  desc=f"Request failed, retrying with with exponential backoff (max_attempts={max_attempts})") as pbar:
            for attempt in pbar:
                try:
                    return await query_model(model_id=model_id, query=query, rate_limiter=rate_limiter)
                except Exception as e:
                    if attempt == max_attempts:
                        raise e
                    await asyncio.sleep(delay + random.uniform(0, 1))
                    delay *= 2


def load_response_file(file: Path | str) -> dict[int, [str, str]]:
    with open(file, "r") as f:
        responses = defaultdict(dict, json.load(f, object_pairs_hook=
        lambda data: {int(k) if k.isdigit() else k: v for k, v in data}))
    return responses


async def _get_responses_from_model(
        queries: list[tuple[int, Query]],
        model: str,
        pbar,  # todo not sure how to type this correctly
        response_dict: dict[int, dict[str, str]],
        max_attempts: int,
        tqdm_pos: int,
        worker_num: int = 1,
        rate_limiter: RateLimiter = None,
) -> None:
    desc = f"Model {model} Worker {worker_num} making queries"
    # TODO fix overlapping progress bars - not worth time rn
    #  also fix bars getting messed up once tasks start to finish
    for i, query in tqdm_asyncio(queries, total=len(queries), position=tqdm_pos, leave=True, desc=desc):
        try:
            response = (await query_model_with_backoff(
                model_id=model,
                query=query,
                max_attempts=max_attempts,
                tqdm_pos=tqdm_pos,
                rate_limiter=rate_limiter,
            )).text
        except Exception as e:  # Backoff strategy failed to get response
            response = ERROR_RESPONSE
        response_dict[i][model] = response
        pbar.update()


async def get_responses_from_models(
        queries: list[Query],
        models: list[str],
        response_file: Path | str,
        max_attempts: int = 5,
        workers_per_model: int = 1,
        requests_per_minute: int = None,
) -> dict[int, dict[str, str]]:
    """
    For each model, send queries that do not yet have a response in response_file, and save response text to that file.
    """
    response_file.parent.mkdir(parents=True, exist_ok=True)
    responses: dict[int, dict[str, str]] = defaultdict(dict)
    solved: dict[str, set[int]] = defaultdict(set)
    # For each model, find which queries have not yet received a response
    if response_file.exists():
        responses = load_response_file(response_file)
        for p_num, response in responses.items():
            for model in models:
                if response.get(model, False) and response[model] != ERROR_RESPONSE:
                    solved[model].add(p_num)

    total = sum(len(queries) - len(solved[model]) for model in models)
    if total == 0:
        return responses
    print(f"{total} / {len(queries) * len(models)} responses have not been received yet.")

    rate_limiter = RateLimiter(requests_per_minute) if requests_per_minute else None

    models = [model for model in models if len(solved[model]) != len(queries)]
    try:
        with tqdm_asyncio(total=total, position=0, desc=f"Making {total} total queries to {len(models)} models") as pbar:
            for m, model in enumerate(models, 0):
                unsolved = [(i, query) for i, query in enumerate(queries) if i not in solved[model]]
                worker_queries = []
                num_workers = min(workers_per_model, len(unsolved))
                queries_per_worker = len(unsolved) // num_workers
                remainder = len(unsolved) % workers_per_model
                start_idx = 0
                for _ in range(remainder):
                    worker_queries.append(unsolved[start_idx:start_idx + queries_per_worker + 1])
                    start_idx += queries_per_worker + 1
                for i in range(remainder, num_workers):
                    worker_queries.append(unsolved[start_idx:start_idx + queries_per_worker])
                    start_idx += queries_per_worker

                # TODO too complex for now, but make sure workers don't stop after they complete task - instead pool tasks and have workers all work on pool
                tasks = []
                for worker in range(num_workers):
                    task = asyncio.create_task(
                        _get_responses_from_model(
                            queries=worker_queries[worker],
                            model=model,
                            response_dict=responses,
                            max_attempts=max_attempts,
                            rate_limiter=rate_limiter,
                            tqdm_pos=m * workers_per_model + worker + 1,
                            worker_num=worker + 1,
                            pbar=pbar,
                        )
                    )
                    tasks.append(task)
                await asyncio.gather(*tasks)
    finally:
        if responses:
            with open(response_file, "w") as f:
                json.dump(responses, f, indent=2)
    return responses
