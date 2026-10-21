from graph.Feature_Graph import *
import random as rd
from sae.SAE_Tools import *
from sae.autoencoder.SAE_Wrapper import HookedSAE
from test_matching.util import *

def _hook_name(sae_name: str, name: str | None):
    return sae_name + "." + str(name)
          
@contextmanager
def setup_cache_sae_hook(
    dict_saes: Dict[str, HookedSAE],
    cache: Dict[str, Tensor],
    use_error_term: bool,
):
    def hook_fn(tens: Tensor, hook: HookPoint, sae: Any, cache: Dict[str, Tensor]) -> None:
        if (
            hook.name == sae_hook_name(sae.cfg.hook_name) or 
            _hook_name(sae.cfg.hook_name, hook.name) == sae_hook_name(sae.cfg.hook_name)
        ):
            cache[sae_hook_name(sae.cfg.hook_name)] = tens.detach()
    
    orig_use_error_term = {}
    try:
        for sae_name, sae in dict_saes.items():
            orig_use_error_term[sae_name] = sae.use_error_term
            sae.use_error_term = use_error_term
            sae.add_hook(
                lambda name: True,
                partial(hook_fn, sae=sae, cache=cache), # type: ignore
                dir="fwd",
            )
        yield
    finally:
        for sae_name, sae in dict_saes.items():
            sae.reset_hooks()
            sae.use_error_term = orig_use_error_term[sae_name]
            
        
@contextmanager
def setup_saes(
    end_node: List[Node],
    end_index: List[Index],
    start_node: List[Node],
    start_index: List[Index],
    cache: Dict[Tuple[Node, Index], Tensor | int],
    count_cache: Dict[Tuple[Node, Index], int],
    clean_cache: Dict[str, Tensor] | ActivationCache,
    corrupt_cache: Dict[str, Tensor] | ActivationCache | None,
    dict_saes: Dict[str, HookedSAE], 
    use_error_term: bool,
    direct_effect: bool,
):      
    track_cache = {}
    def _hook_name(sae_name, hook_name) -> str:
        return f"{sae_name}.{hook_name}"
    
    def hook_fn(tens: Tensor, hook: HookPoint, sae_name: str, direct_effect: bool):
        modified_tens = tens.clone()
        hook_name = _hook_name(sae_name, hook.name)
        
        # Define mapping of index types to their corresponding hook name generators and cache keys
        # End node mappings (for caching differences)
        index_mappings = [(enode, end_id, FeatureIndex, lambda node: sae_hook_name(node.name), True) for enode, end_id in zip(end_node, end_index)]
        # Start node mappings (for applying corruptions)
        index_mappings += [(snode, start_id, FeatureIndex, lambda node: sae_hook_name(node.name), False) for snode, start_id in zip(start_node, start_index)]
        index_mappings += [(snode, start_id, ErrorIndex, lambda node: error_term_name(node.name), False) for snode, start_id in zip(start_node, start_index)]
        
        changed = False
        # Process each mapping
        for node, index, index_type, hook_fn_gen, is_end_node in index_mappings:
            expected_hook = hook_fn_gen(node)
            if hook_name == expected_hook or hook.name == expected_hook:
                if isinstance(index, index_type):
                    changed = True
                    # Determine index dimensions based on index type
                    dim_count = 2 if index_type == ErrorIndex else 3
                    connection_index = ConnectionIndex(return_idx(index.list_index, dim_count))
                    
                    if is_end_node:
                        # Cache the difference for end nodes
                        if (node, index) not in cache:
                            cache[(node, index)] = 0
                            count_cache[(node, index)] = 0
                            
                        new_val = modified_tens[connection_index.as_index].detach()
                        old_val = clean_cache[expected_hook][connection_index.as_index]
                        
                        # check if all start nodes and end node activate
                        track_mask = old_val > 1e-3
                        for (_, _), track_tens in track_cache.items():
                            track_mask &= track_tens
                        
                        cache[(node, index)] += (
                            ((old_val - new_val) / (old_val + 1e-3)) * track_mask
                        ).sum()
                        count_cache[(node, index)] += track_mask.sum().item() # type: ignore
                    else:
                        # track whether the start node activate
                        track_cache[(node, index)] = modified_tens[connection_index.as_index] > 1e-3
                        # Apply corruption for start nodes
                        modified_tens[connection_index.as_index] = (
                            corrupt_cache[expected_hook][connection_index.as_index] if corrupt_cache is not None else 0
                        )
                        
        
        # clean distribution on intermediate nodes
        if direct_effect and not changed:
            try:
                modified_tens = clean_cache[hook.name] # type: ignore
            except KeyError:
                modified_tens = clean_cache[hook_name]
        
        return modified_tens
        
    orig_use_error_term = {}
    try:
        for sae_name, sae in dict_saes.items():
            orig_use_error_term[sae_name] = sae.use_error_term
            sae.use_error_term = use_error_term
            sae.add_hook(
                lambda name: "acts_post" in name.split(".")[-1] or "error" in name.split(".")[-1],
                partial(hook_fn, sae_name=sae_name, direct_effect=direct_effect), # type: ignore
                dir="fwd",
            )
        yield
    finally:
        for sae_name, sae in dict_saes.items():
            sae.reset_hooks()
            sae.use_error_term = orig_use_error_term[sae_name]
            
       
@t.no_grad()
def get_activation_after_patching(
    model: HookedSAETransformer,
    dict_saes: Dict[str, HookedSAE],
    end_node: List[Node],
    end_index: List[Index],
    start_node: List[Node],
    start_index: List[Index],
    data_toks: Tensor,
    use_error_term: bool,
    batch_size: int = 32,
    direct_effect: bool = False,
    verbose: bool = True,
) -> Dict[Tuple[Node, Index], Tensor]:  
    cache = {}
    count_cache = {}
    
    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    for token in tqdm(dataloader, disable=not verbose):
        clean_cache = {}
        with setup_cache_sae_hook(
            dict_saes=dict_saes,
            cache=clean_cache,
            use_error_term=use_error_term,
        ):
            with setup_forward_model_hook(model, list(dict_saes.values()), use_error_term):
                model(token[0])

        with setup_saes(
            end_node=end_node,
            end_index=end_index,
            start_node=start_node,
            start_index=start_index,
            cache=cache,
            count_cache=count_cache,
            clean_cache=clean_cache,
            corrupt_cache=None, # zero patching
            dict_saes=dict_saes, 
            use_error_term=use_error_term, 
            direct_effect=direct_effect,
        ):
            with setup_forward_model_hook(model, list(dict_saes.values()), use_error_term):
                model(token[0])
            
    for key, val in cache.items():
        cache[key] = val / (count_cache[key] if count_cache[key] != 0 else 1) # normalize
        
    return cache