from graph.Feature_Graph import *
import random as rd
from graph_visualizer import *
from sae.SAE_Tools import *
from sae.autoencoder.SAE_Wrapper import HookedSAE

def determine_feature_type(
    graph: Feature_Graph,
    node: Node,
    idx: List[int],
) -> Index:
    d_sae = graph.dict_saes[node.name].cfg.d_sae
    if idx[-1] == d_sae:
        index = ErrorIndex(idx[:-1])
    else:
        index = FeatureIndex(idx)
    return index

def get_scores(
    graph: Feature_Graph,
    sort: bool = False,
    num_sample: int | None = None,
    seed: int = 0,
    abs_score: bool = False,
) -> Dict[Tuple[Node, Index], List[Tuple[Node, Index, float]]]:
    dict_all_list = {}
    for end_node, end_idx in graph.edge_scores.keys():
        for start_node, start_idx in graph.edge_scores[(end_node, end_idx)].keys():
            edge_score = graph.edge_scores[(end_node, end_idx)][(start_node, start_idx)]
            
            for num_active_idx in range(edge_score.indices().shape[1]):
                end_node_idx = edge_score.indices()[:, num_active_idx].tolist()
                end_index = determine_feature_type(graph, end_node, end_node_idx)
                
                if not dict_all_list.get((end_node, end_index), False):
                    dict_all_list[(end_node, end_index)] = []
                    
                # select nonzero edges
                list_indices = (edge_score.values()[num_active_idx] != 0).nonzero().tolist()
                
                # append all edges
                for indices in list_indices:
                    start_index = determine_feature_type(graph, start_node, indices)
                    
                    score = edge_score.values()[(num_active_idx,) + tuple(indices)]
                    if abs_score:
                        score = score.abs()
                        
                    dict_all_list[(end_node, end_index)].append(
                        (
                            start_node,
                            start_index,
                            score.cpu().item()
                        )
                    )
                    
                if sort:
                    dict_all_list[(end_node, end_index)].sort(key=lambda tup: tup[2], reverse=True)
                    
    # sample edges for each end_node
    rd.seed(seed)
    if num_sample is not None:
        for node_and_index, edge_list in dict_all_list.items(): 
            dict_all_list[node_and_index] = rd.sample(edge_list, k=min(num_sample, len(edge_list)))
            
    return dict_all_list


def get_true_effect(
    graph: Feature_Graph,
    clean_token: Tensor, 
    clean_cache: Dict[str, Tensor] | ActivationCache,
    corrupt_cache: Dict[str, Tensor] | ActivationCache,
    use_error_term: bool,
    direct_effect: bool = True,
    sort: bool = False,
    num_sample: int | None = None,
    seed: int = 0,
):
    dict_all_list = {}
    for end_node, end_idx in tqdm(graph.edge_scores.keys()):
        for start_node, start_idx in graph.edge_scores[(end_node, end_idx)].keys():
            edge_score = graph.edge_scores[(end_node, end_idx)][(start_node, start_idx)]
            
            for num_active_idx in range(edge_score.indices().shape[1]):
                end_node_idx = edge_score.indices()[:, num_active_idx].tolist()
                end_index = determine_feature_type(graph, end_node, end_node_idx)
                
                if isinstance(end_index, ErrorIndex):
                    # skip if end_index is ErrorIndex
                    continue
                
                if not dict_all_list.get((end_node, end_index), False):
                    dict_all_list[(end_node, end_index)] = []
                    
                # select nonzero edges
                list_indices = (edge_score.values()[num_active_idx] != 0).nonzero().tolist()
                
                # append all edges
                for indices in list_indices:
                    start_index = determine_feature_type(graph, start_node, indices)
                    
                    patch_effects = get_activation_after_patching(
                        graph.model, 
                        graph.dict_saes, 
                        [end_node], 
                        [end_index], 
                        [start_node], 
                        [start_index], 
                        clean_token, 
                        clean_cache,
                        corrupt_cache,
                        use_error_term,
                        direct_effect
                    )[(end_node, end_index)]
                    
                    dict_all_list[(end_node, end_index)].append(
                        (
                            start_node,
                            start_index,
                            t.mean(patch_effects).abs().cpu().item()
                        )
                    )
                
                if sort:
                    dict_all_list[(end_node, end_index)].sort(key=lambda tup: tup[2], reverse=True)
                    
    # sample edges for each end_node
    rd.seed(seed)
    if num_sample is not None:
        for node_and_index, edge_list in dict_all_list.items(): 
            dict_all_list[node_and_index] = rd.sample(edge_list, k=min(num_sample, len(edge_list)))
            
    return dict_all_list


def get_cluster_true_effect(
    graph_vis: SAEGraphVisualizer,
    clean_token: Tensor, 
    clean_cache: Dict[str, Tensor] | ActivationCache,
    corrupt_cache: Dict[str, Tensor] | ActivationCache,
    use_error_term: bool,
    direct_effect: bool = True,
):  
    assert graph_vis.compressed_graph_json is not None, "Compressed graph is not available in the visualizer."
            
    links = graph_vis.compressed_graph_json['links']
    
    for link in tqdm(links):
        tid = link['target']
        sid = link['source']
        sc = None
        tc = None
        for node in graph_vis.compressed_graph_json['nodes']:
            if node['id'] == sid:
                sc = node['features']
            if node['id'] == tid:
                tc = node['features']
        assert sc is not None and tc is not None, f"Not found for cluster id with source {sid} and target {tid}."
        end_node: List[Node] = [
            ConnectionNode(n["hook"])
            for n in tc
        ]
        end_index: List[Index] = [
            FeatureIndex([n['seq'], n['feat_idx']]) if "feat_idx" in n else ErrorIndex([n['seq']])
            for n in tc
        ]
        start_node: List[Node] = [
            ConnectionNode(n["hook"])
            for n in sc
        ]
        start_index: List[Index] = [
            FeatureIndex([n['seq'], n['feat_idx']]) if "feat_idx" in n else ErrorIndex([n['seq']])
            for n in sc
        ]
        patch_effects = get_activation_after_patching(
            graph_vis.graph.model, 
            graph_vis.graph.dict_saes, 
            end_node,
            end_index, 
            start_node, 
            start_index, 
            clean_token, 
            clean_cache,
            corrupt_cache,
            use_error_term,
            direct_effect
        )
        
        link['true_effect'] = patch_effects

@contextmanager
def setup_saes(
    end_node: List[Node],
    end_index: List[Index],
    start_node: List[Node],
    start_index: List[Index],
    cache: Dict[Tuple[Node, Index], Tensor],
    clean_cache: Dict[str, Tensor] | ActivationCache,
    corrupt_cache: Dict[str, Tensor] | ActivationCache,
    dict_saes: Dict[str, HookedSAE], 
    use_error_term: bool,
    direct_effect: bool,
):      
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
                        cache[(node, index)] = (
                            clean_cache[expected_hook][connection_index.as_index] -
                            modified_tens[connection_index.as_index].detach()
                            
                        )
                    else:
                        # Apply corruption for start nodes
                        modified_tens[connection_index.as_index] = corrupt_cache[expected_hook][connection_index.as_index]
        
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
            sae.reset_saes()
            sae.reset_hooks()
            sae.use_error_term = orig_use_error_term[sae_name]
            
@contextmanager
def setup_forward_model_hook(
    model: HookedSAETransformer, list_saes: List[HookedSAE], use_error_term: bool
):
    input_hooks_trans = []
    output_hooks_trans = []
    transcoders = []
    input_hooks_cross = []
    output_hooks_cross = []
    crosscoders = []
    non_cross_or_trans = []
    for sae in list_saes:
        if getattr(sae, "input_hook", False) and getattr(sae, "output_hook", False):
            input_hooks_trans.append(sae.input_hook)
            output_hooks_trans.append(sae.output_hook)
            transcoders.append(sae)
        elif getattr(sae, 'input_hook', False) and getattr(sae, 'output_hooks', False):
            input_hooks_cross.append(sae.input_hook)
            output_hooks_cross.append(sae.output_hooks[0]) # type: ignore
            crosscoders.append(sae)
        else:
            non_cross_or_trans.append(sae)
            
    # Transcoder
    trans_cache = {}
    # Hook function at transcoder input: caches the activations before transcoder
    def hook_transcoder_input(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        trans_cache[transcoder_idx] = activations.clone()

    # Hook function at transcoder output: replaces activations with transcoder output
    def hook_transcoder_output(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        trans_input = trans_cache[transcoder_idx]
        return transcoders[transcoder_idx]((trans_input, activations))
    
    # Crosscoder
    cross_cache = {}
    # Hook function at transcoder input: caches the activations before transcoder
    def hook_crosscoder_input(activations: Tensor, hook: HookPoint, crosscoder_idx: int):
        cross_cache[crosscoder_idx] = activations.clone()
    
    all_cross_recons = [0.0 for _ in range(len(crosscoders))]
    def hook_crosscoder_output(activations: Tensor, hook: HookPoint, crosscoder_idx: int):
        recons, cross_recons = crosscoders[crosscoder_idx].forward_crosscoder(
            (cross_cache[crosscoder_idx], activations, all_cross_recons[crosscoder_idx])
        )
        for j in range(crosscoder_idx+1, model.cfg.n_layers):
            all_cross_recons[j] += cross_recons[j-crosscoder_idx-1]
        
        return recons
    
    fwd_hooks = []
    for i in range(len(transcoders)):
        fwd_hooks.append((input_hooks_trans[i], partial(hook_transcoder_input, transcoder_idx=i)))
        fwd_hooks.append((output_hooks_trans[i], partial(hook_transcoder_output, transcoder_idx=i)))
    for i in range(len(crosscoders)):
        fwd_hooks.append((input_hooks_cross[i], partial(hook_crosscoder_input, crosscoder_idx=i)))
        fwd_hooks.append((output_hooks_cross[i], partial(hook_crosscoder_output, crosscoder_idx=i)))
        
    try: 
        for sae in non_cross_or_trans:
            model.add_sae(sae, use_error_term)
        for hook, func in fwd_hooks:
            model.add_hook(hook, func, dir="fwd")
        
        yield
    finally:
        model.reset_saes()
        model.reset_hooks()
            
def get_activation_after_patching(
    model: HookedSAETransformer,
    dict_saes: Dict[str, HookedSAE],
    end_node: List[Node],
    end_index: List[Index],
    start_node: List[Node],
    start_index: List[Index],
    clean_token: Tensor,
    clean_cache: Dict[str, Tensor] | ActivationCache,
    corrupt_cache: Dict[str, Tensor] | ActivationCache,
    use_error_term: bool,
    direct_effect: bool = True,
) -> Dict[Tuple[Node, Index], Tensor]:  
    cache = {}
    with setup_saes(
        end_node=end_node,
        end_index=end_index,
        start_node=start_node,
        start_index=start_index,
        cache=cache,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        dict_saes=dict_saes, 
        use_error_term=use_error_term, 
        direct_effect=direct_effect,
    ):
        with setup_forward_model_hook(model, list(dict_saes.values()), use_error_term):
            model(clean_token)
            
    return cache
