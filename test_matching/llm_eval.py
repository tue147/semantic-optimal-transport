import argparse
import asyncio
import gc
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal, TypeAlias

import torch
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError, AsyncOpenAI, AuthenticationError
from sae_lens import SAE
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae.SAE_Tools import *
from graph_visualizer import *
import uuid
from .util import *

def get_eval_uuid():
    return str(uuid.uuid4())

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


def display_messages(messages: Messages) -> str:
    return tabulate(
        [m.values() for m in messages], tablefmt="simple_grid", maxcolwidths=[None, 120]
    )


def str_bool(b: bool) -> str:
    return "Y" if b else ""


def escape_slash(s: str) -> str:
    return s.replace("/", "_")


class Example:
    """
    Data for a single example sequence.
    """

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model: HookedTransformer,
    ):
        self.toks = toks
        self.str_toks = model.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(
            self.toks_are_active
        )  # this is what we predict in the scoring phase

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)  # type: ignore
            )
            .replace("�", "")
            .replace("\n", "↵")
            # .replace(">><<", "")
        )
        
class Examples:
    """
    Data for multiple example sequences. Includes methods for shuffling seuqences, and displaying them.
    """

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(
                self.examples, key=lambda x: max(x.acts), reverse=True
            )

    def display(self, predictions: list[int] | None = None) -> str:
        """
        Displays the list of sequences. If `predictions` is provided, then it'll include a column for both "is_active"
        and these predictions of whether it's active. If not, then neither of those columns will be included.
        """
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        str_bool(ex.is_active),
                        str_bool(i + 1 in predictions),
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]
    
    
def get_generation_prompts(
    prompt_feat_a_data: list[Examples], 
    prompt_feat_b_data: list[Examples], 
) -> Messages:
    examples_str_feature_a = []
    for i, examples in enumerate(prompt_feat_a_data):
        examples_str_feature_a.append(f"\nFeature {i+1}: \n"+ format_feature_examples(examples))
    feature_a_header = f'''
    FEATURE 1:
    {"".join(examples_str_feature_a)}
    '''
    
    examples_str_feature_b = []
    
    for i, examples in enumerate(prompt_feat_b_data):
        examples_str_feature_b.append(f"\nFeature {i+1}: \n"+ format_feature_examples(examples))
        
    feature_b_header = f'''
    FEATURE 2:
    {"".join(examples_str_feature_b)}
    '''
    
    SYSTEM_PROMPT = """
    ### TASK:
    We're studying a pair of features in a neural network.
    You will rate whether the feature in the pairs are similar, 3 possible outputs are: SIMILAR / MAYBE / DIFFERENT.
    
    ### MEANING OF FEATURE and FEATURE CAPTIONING:
    Each feature activates on some particular word/words/substring/concept in a short document. 
    The activating words in each document are indicated with << ... >>. 
    We will give you a list of documents on which the feature activates, in order from most strongly activating to least strongly activating. 
    Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is activating on. 
    Try not to be overly specific in your explanation. 
    Note that some features will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. 
    Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the feature to activate). 
    Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. 
    Keep the explanation as short and simple as possible, limited to 20 words or less. 
    Omit punctuation and formatting. 
    You should avoid giving long lists of words.
    Some examples: "This feature activates on the word 'knows' in rhetorical questions", and "This feature activates on verbs related to decision-making and preferences", and "This feature activates on the substring 'Ent' at the start of words", and "This feature activates on text about government economic policy".
    
    ### INPUT FORMAT:
    "
    FEATURE 1: 
    1. (example 1)
    2, ....
    
    FEATURE 2: 
    ...
    "
    
    ### STEPS:
    1, First you need to caption the meaning of FEATURES.
    2, Determine if the two features have similar meaning or not. If any feature has "NO EXAMPLE.", you must output DIFFERENT.
    
    ### OUTPUT FORMAT:
    "
        FEATURE CAPTION:
    
    Feature 1: (caption)
    Feature 2: (caption)

    =====================
        ANSWER: SIMILAR / MAYBE / DIFFERENT
    "
    """
    
    USER_PROMPT = (
        f"""The activating documents are given below:\n\n{feature_a_header + feature_b_header}"""
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    
def format_feature_examples(feature_examples: Examples) -> str:
    if len(feature_examples) == 0:
        return "NO EXAMPLE."

    examples_as_str = "\n".join(
        [
            f"{i + 1}. {ex.to_str(mark_toks=True)}"
            for i, ex in enumerate(feature_examples)
        ]
    )
    return examples_as_str
'''
pseudocode

for each cluster:
    skip cluster with 1 feature
    sample n/2 or max number of feature in the cluster
    sample n/2 or max number of feature in the cluster and not in the cluster
    get feature info
    prompt llm to predict which feature should be in the cluster
    measure the accuracy
    return the result
'''

def get_feature_examples(
    indices: Tensor,
    values: Tensor, 
    data_toks: Tensor,
    model: HookedTransformer,
    feats: List[int],
    n_intervals: int = 3,
    k_per_interval: int = 3,
    buffer: int = 10, 
) -> List[Examples]:
    prompt_feature_data = []
    for feat_idx in feats:
        all_data = fetch_feature_activation_intervals(
            model, feat_idx, values, indices, data_toks, n_intervals, k_per_interval, buffer
        )
        feat_data = []
        for interval_data in all_data['intervals']:
            examples = interval_data['examples']
            for example in examples:
                feat_data.append(
                    Example(
                        # raw ids straight from the cache: re-tokenizing the decoded strings is a
                        # lossy round-trip (a str token can split into >1 id, e.g. under gpt2)
                        toks=example['short']['token_ids'],
                        acts=example['short']['activations'],
                        act_threshold=1e-3,
                        model=model
                    )
                )
        prompt_feature_data.append(Examples(feat_data, shuffle=False))
    return prompt_feature_data

def build_completion_kwargs(judge: dict, messages: Messages, max_tokens: int) -> dict:
    """Per-judge kwargs for chat.completions.create.

    gpt5-* are reasoning models: they take `max_completion_tokens` + `reasoning_effort` and reject
    `max_tokens`. The UF-gateway chat models take the plain `max_tokens`.
    """
    kwargs = dict(model=judge["model"], messages=messages, stream=False)
    if judge.get("style") == "reasoning":
        kwargs["max_completion_tokens"] = max_tokens
        if judge.get("reasoning_effort"):
            kwargs["reasoning_effort"] = judge["reasoning_effort"]
    else:
        kwargs["max_tokens"] = max_tokens
    return kwargs


async def get_api_response(
    client: AsyncOpenAI,
    messages: Messages,
    req_sem: asyncio.Semaphore,
    judge: dict,
    max_tokens: int = 10000,
    max_retries: int = 12  # was 5 (~8 s total): too short when several jobs share the UF gateway
) -> str:
    attempt = 0
    base_backoff = 0.5
    while True:
        attempt += 1
        try:
            async with req_sem:
                result = await client.chat.completions.create(
                    **build_completion_kwargs(judge, messages, max_tokens),  # type: ignore
                )
            content = result.choices[0].message.content
            return content.strip() if content else ""  # reasoning models can return empty content

        # AuthenticationError: the UF gateway sporadically 401s a valid key (2026-10-14 16:28, both lanes)
        except (RateLimitError, APITimeoutError, APIConnectionError, AuthenticationError) as e:
            if attempt >= max_retries:
                raise
            await asyncio.sleep(min(60.0, base_backoff * (2 ** (attempt - 1))) * (0.5 + random.random()))
            
        except APIError as e:
            if getattr(e, "status_code", 500) >= 500 and attempt < max_retries:
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)) * (0.5 + random.random()))
            else:
                raise

def parse_matching_predictions(response: str) -> int:
    """Parses the 'ANSWER:' block from the LLM to extract Yes/No predictions."""
    # try:
    score = 1  # fall back if the response is empty or has no parseable ANSWER block
    answer_block = response.split("ANSWER")[-1].strip()

    parts = answer_block.split(':')
    if len(parts) > 1:
        ans = parts[1].strip().lower()
        if 'similar' in ans or '\"similar\"' in ans:
            score = 3
        elif 'maybe' in ans or '\"maybe\"' in ans:
            score = 2
        elif 'different' in ans or '\"different\"' in ans:
            score = 1
        else:
            score = 1 # fall back
        
    return score
    # except Exception as e:
    #     print(f"Failed to parse response: {e}")
    #     return 1

@t.no_grad()
async def llm_matching_eval(
    feat_pairs: List[Tuple[int, int]], # feat_a, feat_b
    model: HookedTransformer,
    data_toks: Tensor,
    indices_a: Tensor,
    values_a: Tensor,
    indices_b: Tensor,
    values_b: Tensor,
    judge: dict,
    n_intervals: int = 3,
    k_per_interval: int = 3,
    buffer: int = 10,
    max_parallel_requests: int = 4  # UF gateway caps a key at 10 in-flight requests; 2 jobs x 4 fit
):
    client = AsyncOpenAI(
        api_key=judge["api_key"],
        base_url=judge.get("base_url"),  # None -> OpenAI's public API for the gpt5-* judges
        timeout=judge.get("timeout", 45.0),
        max_retries=0,
    )
    req_sem = asyncio.Semaphore(max_parallel_requests)
    
    tasks = []

    for feat_a, feat_b in tqdm(feat_pairs, desc="Preparing Data"):
        prompt_feat_a_data = get_feature_examples(
            indices_a, values_a, data_toks, model, [feat_a], n_intervals, k_per_interval, buffer
        )
        prompt_feat_b_data = get_feature_examples(
            indices_b, values_b, data_toks, model, [feat_b], n_intervals, k_per_interval, buffer
        )
        
        # Get messages AND track ground truth
        messages = get_generation_prompts(
            prompt_feat_a_data, prompt_feat_b_data
        )
        
        # Append the API call to our async task lis
        tasks.append(get_api_response(client, messages, req_sem, judge))

    print(f"Executing LLM evaluation for {len(tasks)} clusters concurrently...")
    
    # Gather responses
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    assert len(responses) == len(feat_pairs)
    total_score = 0
    final_results = []
    
    # Evaluate Results
    for i, response in enumerate(responses):
        score = parse_matching_predictions(response) # type: ignore
        
        total_score += score
        
        final_results.append({
            "feature_pair": feat_pairs[i],
            "score": score,
            "raw_response": response
        })
        

    overall_score = total_score / len(responses) if len(responses) > 0 else 0
    
    print("\n--- Final Results ---")
    print(f"Total Clusters Evaluated: {len(final_results)}")
    print(f"Overall Score: {overall_score}")
    
    return {
        "overall_score": overall_score,
        "total_score": total_score,
        "total_predictions": len(responses),
        "matching_details": final_results
    }