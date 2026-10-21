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
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError, AsyncOpenAI
from sae_lens import SAE
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae.SAE_Tools import *
from graph_visualizer import *
import uuid

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
    prompt_feature_data: list[Examples], 
    test_feature_data: list[Examples], 
    rand_feature_data: list[Examples],
) -> tuple[Messages, list[bool]]:
    examples_str_prompt_feature = []
    for i, examples in enumerate(prompt_feature_data):
        examples_str_prompt_feature.append(f"\nFeature {i+1}: \n"+ format_feature_examples(examples))
    prompt_feature_header = f'''
    FEATURE IN CLUSTER:
    {"".join(examples_str_prompt_feature)}
    '''
    
    # Combine test and rand features, keeping track of ground truth (True = in cluster, False = outside cluster)
    combined_test_features = [(data, True) for data in test_feature_data] + [(data, False) for data in rand_feature_data]
    random.shuffle(combined_test_features)
    
    examples_str_all_test_feature = []
    ground_truth = []
    
    for i, (examples, is_in_cluster) in enumerate(combined_test_features):
        examples_str_all_test_feature.append(f"\nFeature {i+1}: \n"+ format_feature_examples(examples))
        ground_truth.append(is_in_cluster)
        
    test_feature_header = f'''
    FEATURE TO PREDICT:
    {"".join(examples_str_all_test_feature)}
    '''
    
    SYSTEM_PROMPT = """
    TASK:
    We're studying a cluster of features in a neural network.
    Each cluster contains a few features, each with similar high level meaning (the meaning of the feature is defined below).
    Given a number of features in the cluster, you must predict another number of features if they are in or not in the cluster based on whether their meaning is relevant to the cluster.
    
    MEANING OF FEATURE and FEATURE CAPTIONING:
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
    
    INPUT FORMAT:
    "
        FEATURE IN CLUSTER:
    
    Feature 1: 
    1. (example 1)
    2, ....
    Feature 2: 
    ...

        FEATURE TO PREDICT:
        
    Feature 1: 
    1. (example 1)
    2, ....
    Feature 2: 
    ...
    "
    
    STEPS:
    1, First you need to caption the meaning of FEATURE IN CLUSTER.
    2, Then you create a brief summarization of the highlevel meaning of the features in the cluster. Similar to how you caption the meaning of feature, it should be generalizable and not overly specific.
    3, Then you caption the meaning of FEATURE TO PREDICT.
    4, Determine the test features whether they should be in the cluster or not.
    
    OUTPUT FORMAT:
    "
        FEATURE IN CLUSTER:
    
    Feature 1: (caption)
    Feature 2: (caption)
    ...
    Cluster meaning: (caption)
    
        FEATURE TO PREDICT:
        
    Feature 1: (caption)
    Feature 2: (caption)
    ...
    =====================
        ANSWER:
    Feature 1: Yes / No
    Feature 2: Yes / No
    ...
    "
    """
    
    USER_PROMPT = (
        f"""The activating documents are given below:\n\n{prompt_feature_header + test_feature_header}"""
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ], ground_truth
    
def format_feature_examples(feature_examples: Examples) -> str:
    assert len(feature_examples) > 0, "No generation examples found"

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
    graph_vis: SAEGraphVisualizer,
    features: List[Dict],
    n_intervals: int = 3,
    k_per_interval: int = 3,
    buffer: int = 10, 
) -> List[Examples]:
    prompt_feature_data = []
    for pf in features:
        feat_idx = pf['feat_idx']
        hook = pf['hook']
        values, indices = graph_vis.full_data_dict[hook] # type: ignore
        
        all_data = fetch_feature_activation_intervals(
            graph_vis.model, feat_idx, values, indices, graph_vis.data_toks, n_intervals, k_per_interval, buffer
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
                        model=graph_vis.model
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

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt >= max_retries:
                raise
            await asyncio.sleep(min(60.0, base_backoff * (2 ** (attempt - 1))) * (0.5 + random.random()))
            
        except APIError as e:
            if getattr(e, "status_code", 500) >= 500 and attempt < max_retries:
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)) * (0.5 + random.random()))
            else:
                raise

def parse_cluster_predictions(response: str, expected_count: int) -> list[bool]:
    """Parses the 'ANSWER:' block from the LLM to extract Yes/No predictions."""
    # try:
    answer_block = response.split("ANSWER:")[-1].strip()
    lines = [line.strip() for line in answer_block.split('\n') if line.strip().lower().startswith('feature')]
    
    predictions = []
    for line in lines:
        parts = line.split(':')
        if len(parts) > 1:
            ans = parts[1].strip().lower()
            if 'yes' in ans:
                predictions.append(True)
            elif 'no' in ans:
                predictions.append(False)
    
    # In case the LLM formatting breaks, pad with False to avoid list size mismatch during scoring
    if len(predictions) < expected_count:
        predictions.extend([False] * (expected_count - len(predictions)))
        
    return predictions[:expected_count]
    # except Exception as e:
    #     print(f"Failed to parse response: {e}")
    #     return [False] * expected_count

@t.no_grad()
async def llm_clustering_eval(
    graph_vis: SAEGraphVisualizer,
    judge: dict,
    n_cluster_feature: int = 3,
    n_rand_feature: int = 3,
    n_intervals: int = 3,
    k_per_interval: int = 3,
    buffer: int = 10,
    max_parallel_requests: int = 5
):
    graph_data = graph_vis.build_graph_json()
    feature_nodes = [n for n in graph_data['nodes'] if n['type'] == 'feature']
    
    # Map nodes to clusters and filter valid clustered nodes
    node_to_cluster = {}
    for n in graph_data['nodes']:
        if n.get('cluster') is not None:
            if n['cluster'] not in node_to_cluster:
                node_to_cluster[n['cluster']] = []
            node_to_cluster[n['cluster']].append(n)
            
    if not node_to_cluster:
        print("No clustered nodes found. Run clustering first.")
        return None

    client = AsyncOpenAI(
        api_key=judge["api_key"],
        base_url=judge.get("base_url"),  # None -> OpenAI's public API for the gpt5-* judges
        timeout=judge.get("timeout", 100.0),
        max_retries=0,
    )
    req_sem = asyncio.Semaphore(max_parallel_requests)
    
    tasks = []
    cluster_metadata = []

    for cluster_id, nodes in tqdm(node_to_cluster.items(), desc="Preparing Data"):
        # Ensure we have enough features to form a test
        n_prompt_feature_cluster = min(len(nodes) // 2, n_cluster_feature)
        if n_prompt_feature_cluster == 0 and len(nodes) > 1:
            n_prompt_feature_cluster = 1 
            
        if len(nodes) < 2: 
            continue # Skip clusters with only 1 feature
        
        prompt_feature = nodes[:n_prompt_feature_cluster]
        test_feature = nodes[n_prompt_feature_cluster : n_prompt_feature_cluster + n_cluster_feature]
        
        feature_from_outside = [n for n in feature_nodes if n['cluster'] != cluster_id]
        rand_feature = random.choices(feature_from_outside, k=min(n_rand_feature, len(feature_from_outside)))
        
        prompt_feature_data = get_feature_examples(graph_vis, prompt_feature, n_intervals, k_per_interval, buffer)
        test_feature_data = get_feature_examples(graph_vis, test_feature, n_intervals, k_per_interval, buffer)
        rand_feature_data = get_feature_examples(graph_vis, rand_feature, n_intervals, k_per_interval, buffer)
        
        # Get messages AND track ground truth
        messages, ground_truth = get_generation_prompts(
            prompt_feature_data, test_feature_data, rand_feature_data
        )
        
        cluster_metadata.append({
            "cluster_id": cluster_id,
            "ground_truth": ground_truth,
            "n_predict": len(ground_truth)
        })
        
        # Append the API call to our async task list
        tasks.append(get_api_response(client, messages, req_sem, judge))

    print(f"Executing LLM evaluation for {len(tasks)} clusters concurrently...")
    
    # Gather responses
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_correct = 0
    total_predictions = 0
    final_results = []
    
    # Evaluate Results
    for metadata, response in zip(cluster_metadata, responses):
        cluster_id = metadata["cluster_id"]
        ground_truth = metadata["ground_truth"]
        n_predict = metadata["n_predict"]
        
        if isinstance(response, Exception):
            print(f"Cluster {cluster_id} failed with error: {response}")
            continue

        predictions = parse_cluster_predictions(response, n_predict) # type: ignore
        
        correct_count = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct_count / n_predict if n_predict > 0 else 0
        
        final_results.append({
            "cluster_id": cluster_id,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": n_predict,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "raw_response": response
        })
        
        total_correct += correct_count
        total_predictions += n_predict

    # A circuit with no supernode of >= 2 nodes grades nothing: its accuracy is undefined (NaN), not 0.
    # Callers skip it when averaging (the published Table 2 means are over gradable circuits only).
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else float("nan")
    
    print("\n--- Final Results ---")
    print(f"Total Clusters Evaluated: {len(final_results)}")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_predictions})")
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_predictions": total_predictions,
        "cluster_details": final_results
    }
            