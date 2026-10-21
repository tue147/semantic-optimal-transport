import torch as t
from torch import Tensor
import numpy as np
import json
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Any, Dict, List
from sae_lens import HookedSAETransformer
from llm_interpret_circuit_util import call_llm_for_clustering

@t.no_grad()
def match_intra_sae_features_llm(
    data_toks: Tensor,
    model: HookedSAETransformer,
    sae: Any,
    feature_pool: t.Tensor,
    api_key: str,  # Replaced llm_callable with api_key
    k: int = 5,
    batch_size: int = 8,
    chunk_size: int = 20,
    num_top_contexts: int = 10,
    context_window: int = 6,
    activation_threshold: float = 1e-3,
) -> Dict[int, List[Dict[str, float]]]:
    """
    Matches features by leveraging an LLM to caption feature activations and then 
    finding the most semantically similar features based on those captions.
    """
    
    # ==========================================
    # Phase 1: Extract Top Contexts & Captioning
    # ==========================================
    
    feat_list = feature_pool.tolist()
    feature_pool_device = feature_pool.to(model.cfg.device)
    
    feature_activations = {
        feat_idx: {'toks': [], 'weights': []} for feat_idx in feat_list
    }
    
    global_tok_offset = 0
    seq_len = data_toks.shape[1]
    sae.reset_hooks()
    
    def hook_sae_fn(inputs: Tensor, hook: Any) -> None:
        acts = inputs.detach().view(-1, inputs.shape[-1])
        pool_acts = acts[:, feature_pool_device]
        
        for i, feat_idx in enumerate(feat_list):
            f_acts = pool_acts[:, i]
            active_mask = f_acts > activation_threshold
            active_local_toks = active_mask.nonzero(as_tuple=True)[0]
            
            if len(active_local_toks) > 0:
                feature_activations[feat_idx]['toks'].append(active_local_toks.cpu() + global_tok_offset)
                feature_activations[feat_idx]['weights'].append(f_acts[active_local_toks].cpu())

    sae.add_hook(lambda n: "acts_post" in n, hook_sae_fn, dir="fwd")
    
    def trigger_sae(tens: Tensor, hook: Any):
        _ = sae(tens)
        return tens
    
    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    
    for (batch_toks,) in tqdm(dataloader, desc="Extracting LLM Context Hooks"):
        toks = batch_toks.to(model.cfg.device)
        flat_bs = toks.shape[0] * toks.shape[1]
        
        with model.hooks(fwd_hooks=[
            (sae.cfg.hook_name, trigger_sae) 
        ]):
            model(toks)
            
        global_tok_offset += flat_bs
        
    sae.reset_hooks()
    
    # Generate Captions via LLM
    captions = {}
    phase_1_system = "You are an AI interpretability assistant analyzing neural network features. You must respond in valid JSON."
    
    for feat_idx in tqdm(feat_list, desc="Generating Captions"):
        if len(feature_activations[feat_idx]['weights']) == 0:
            captions[feat_idx] = "Dead feature with no activations."
            continue
            
        f_toks = t.cat(feature_activations[feat_idx]['toks'])
        f_weights = t.cat(feature_activations[feat_idx]['weights'])
        
        # Get top contexts
        if len(f_weights) > num_top_contexts:
            top_weights, top_indices = t.topk(f_weights, num_top_contexts)
            top_global_toks = f_toks[top_indices]
        else:
            top_weights = f_weights
            top_global_toks = f_toks
            
        # Decode the context strings directly from the main data_toks tensor
        context_lines = []
        for weight, global_idx in zip(top_weights, top_global_toks):
            b_idx = (global_idx // seq_len).item()
            s_idx = (global_idx % seq_len).item()
            
            start = max(0, s_idx - context_window)
            end = min(seq_len, s_idx + context_window + 1)
            
            context_toks = data_toks[b_idx, start:end] # type: ignore
            context_str = model.tokenizer.decode(context_toks).replace('\n', '\\n') # type: ignore
            context_lines.append(f"- Act: {weight.item():.3f} | Text: '{context_str}'")
        
        context_str_joined = "\n".join(context_lines)
        
        # Force JSON constraint in Phase 1
        prompt_1 = (
            f"Here are the highest activating text snippets for Feature {feat_idx}:\n{context_str_joined}\n\n"
            f"Describe what concept, pattern, or semantic meaning this feature activates on in EXACTLY ONE SHORT SENTENCE. "
            f"Do not add any conversational filler.\n"
            f"Output your response strictly as a JSON object with a single key 'caption'."
        )
        
        # Call LLM & Parse
        raw_response = call_llm_for_clustering(api_key, phase_1_system, prompt_1)
        try:
            caption_data = json.loads(raw_response)
            captions[feat_idx] = caption_data.get("caption", "").strip()
        except json.JSONDecodeError:
            captions[feat_idx] = "Failed to generate caption due to formatting error."

    # ==========================================
    # Phase 2: Chunked LLM Matching
    # ==========================================
    
    # Prepare the full catalog string
    full_catalog_lines = [f"Feature {f}: {cap}" for f, cap in captions.items()]
    full_catalog_str = "\n".join(full_catalog_lines)
    
    feature_list = feature_pool.tolist()
    results = {}
    
    phase_2_system = "You are an AI interpretability assistant. You must respond in valid JSON."
    
    # Process target features in chunks
    for i in tqdm(range(0, len(feature_list), chunk_size), desc="LLM Matching"):
        chunk = feature_list[i:i + chunk_size]
        target_str = ", ".join([str(f) for f in chunk])
        
        prompt_2 = (
            f"You are given a full catalog of features and their single-sentence descriptions.\n\n"
            f"FULL FEATURE CATALOG:\n{full_catalog_str}\n\n"
            f"Your task is to find the top {k} most semantically related features for the following target features.\n"
            f"TARGET FEATURES TO MATCH: {target_str}\n\n"
            f"RULES:\n"
            f"1. Return EXACTLY {k} matches for each target feature.\n"
            f"2. DO NOT match a target feature to itself.\n"
            f"3. Output your response STRICTLY as a JSON object where keys are the target feature IDs (as strings) "
            f"and values are lists of {k} integer feature IDs."
        )
        
        raw_response = call_llm_for_clustering(api_key, phase_2_system, prompt_2)
        
        # No need to manually strip markdown; json_object format blocks markdown wrappers inherently.
        try:
            llm_matches = json.loads(raw_response.strip())
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON for chunk starting at {chunk[0]}. Falling back entirely to random.")
            llm_matches = {}

        # Parse matches and apply fallback logic
        for f in chunk:
            raw_matches = llm_matches.get(str(f), [])
            
            if not isinstance(raw_matches, list): 
                raw_matches = []
                
            clean_matches = []
            for m in raw_matches:
                try:
                    clean_matches.append(int(m))
                except (ValueError, TypeError):
                    continue
                    
            matches = [m for m in clean_matches if m != f and m in feature_list]
            
            # Fallback: Fill with random distinct features if we don't have enough valid matches
            available = [x for x in feature_list if x != f and x not in matches]
            while len(matches) < k and len(available) > 0:
                random_feat = random.choice(available)
                matches.append(random_feat)
                available.remove(random_feat)
                
            # Truncate if the LLM hallucinated more than k
            matches = matches[:k]
            
            # Format output to match standard dictionary structure
            results[f] = [{'feature': m, 'llm_matched': True} for m in matches]
            
    return results