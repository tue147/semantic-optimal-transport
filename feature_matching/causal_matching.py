from graph.Feature_Graph import *
from torch.utils.data import DataLoader

@contextmanager
def _setup_fwd_bwd_edge_grad_sae_hook(
    dict_saes: Dict[str, Any],
    bwd_cache: Dict[str, Tensor],
    to_bwd_cache: Dict[str, Tensor],
    fwd_cache: Dict[str, Tensor],
    hook_position_b: str,
    hook_position_a: str,
    direct_effect: bool = True,
    use_error_term: bool = True,
):
    def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str, bwd_cache: Dict) -> None:
        if hook.name == sae_hook_name(hook_position_b):
            # only store the gradient at the start position
            bwd_cache[sae_hook_name(hook_position_b)] = grad.detach()
            
        elif hook.name == output_hook_name(hook_position_b):
            # we have: output = recon + stop_grad(error_term)
            # so, the TRUE error_grad (if not stop_grad) is output_grad 
            # due to: output = recon + error_term
            if use_error_term:
                # only store the gradient at the start position
                bwd_cache[error_term_name(hook_position_b)] = grad.detach()

        # IMPORTANT NOTE for reproducibility:
        # we zero grad of intermediate components
        # but zero grad the output hook of SAE will ALSO ZERO GRAD the resid mid / post --> no downstream grads
        # so, we instead zero grad of the input hook of SAE 
        elif hook.name == input_hook_name(sae_name):
            if hook.name != input_hook_name(hook_position_a) and direct_effect:
                grad.zero_()

    def hook_sae_fwd(act: Tensor, hook: HookPoint, to_bwd_cache: Dict) -> Tensor:
        if hook.name == sae_hook_name(hook_position_a):
            # store activation for backward later
            to_bwd_cache[sae_hook_name(hook_position_a)] = act
            
        elif error_term_name(hook_position_a) == hook.name and use_error_term:
            # store activation for backward later
            to_bwd_cache[error_term_name(hook_position_a)] = act
            
        elif hook.name == sae_hook_name(hook_position_b):
            # store activation for backward later
            fwd_cache[sae_hook_name(hook_position_b)] = act.detach()
            
        elif error_term_name(hook_position_b) == hook.name and use_error_term:
            # store activation for backward later
            fwd_cache[error_term_name(hook_position_b)] = act.detach()
                
        return act
    
    try:
        with _setup_error_term(dict_saes, use_error_term):
            for sae_name, sae in dict_saes.items():
                sae.add_hook(
                    lambda name: True,
                    partial(hook_sae_fwd, to_bwd_cache=to_bwd_cache),
                    dir="fwd",
                )
                sae.add_hook(
                    lambda name: True,
                    partial(hook_sae_bwd, sae_name=sae_name, bwd_cache=bwd_cache),
                    dir="bwd",
                )
            yield
    finally:
        for sae in dict_saes.values():
            sae.reset_hooks()

# @contextmanager
# def _setup_fwd_sae_hook(
#     dict_saes: Dict[str, Any],
#     fwd_cache: Dict[str, Tensor],
#     corrupt_cache: ActivationCache | Dict[str, Tensor] | None, 
#     patch_deleted_comp: bool = False,
#     use_error_term: bool = True,
# ):    
#     def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str) -> Tensor:
#         if hook.name == sae_hook_name(sae_name):
#             if patch_deleted_comp and corrupt_cache is not None:
#                 act_mask = self.nodes[(sae_name, (None,))].act == 0 # type: ignore
#                 act[:, act_mask] = corrupt_cache[sae_hook_name(sae_name)][:, act_mask]
#             fwd_cache[sae_hook_name(sae_name)] = act.detach()
            
#         elif hook.name == error_term_name(sae_name) and use_error_term:
#             if patch_deleted_comp and corrupt_cache is not None:
#                 resc_mask = self.nodes[(sae_name, (None,))].resc == 0 # type: ignore
#                 act[:, resc_mask] = corrupt_cache[error_term_name(sae_name)][:, resc_mask]
#             fwd_cache[error_term_name(sae_name)] = act.detach()
            
#         return act
    
#     try:
#         with _setup_error_term(dict_saes, use_error_term):
#             for sae_name, sae in dict_saes.items():
#                 sae.add_hook(
#                     lambda name: True,
#                     partial(hook_sae_fwd, sae_name=sae_name),
#                     dir="fwd",
#                 )
#             yield
#     finally:
#         for sae in dict_saes.values():
#             sae.reset_hooks()
            
@contextmanager
def _setup_forward_model_hook(model: HookedSAETransformer, dict_saes: Dict[str, Any], use_error_term: bool = True):
    use_error_term = use_error_term if use_error_term is not None else use_error_term
    try: 
        for sae_name, sae in dict_saes.items():
            model.add_sae(sae, use_error_term)
            
        yield
    finally:
        model.reset_saes()
            
@contextmanager  
def _setup_error_term(dict_saes: Dict[str, Any], use_error_term: bool = True):
    if use_error_term is None:
        use_error_term = self.use_error_term
    
    orig_use_error_term = {}
    try: 
        for sae_name, sae in dict_saes.items():
            sae.use_error_term = use_error_term
            orig_use_error_term[sae_name] = sae.use_error_term
        yield
    finally:
        for sae_name, sae in dict_saes.items():
            sae.use_error_term = orig_use_error_term[sae_name]   

@contextmanager
def _detach_error_term(dict_saes: Dict[str, Any], detach: bool, sae_name: str | None = None):
    orig_detach_error_term = {}
    orig_disable_error_grad = {}
    try:
        for name, sae in dict_saes.items():
            if sae_name is None or sae_name == name:
                orig_detach_error_term[name] = sae.detach_error_term
                sae.detach_error_term = detach
            else:
                orig_detach_error_term[name] = sae.detach_error_term
                sae.detach_error_term = True # default detach error term
            
            orig_disable_error_grad[name] = sae.disable_error_grad
            sae.disable_error_grad = False # allow grad flows through error hook
        yield
    finally:
        for name, sae in dict_saes.items():
            sae.detach_error_term = orig_detach_error_term[name]
            sae.disable_error_grad = orig_disable_error_grad[name]

def model_setup(model: HookedSAETransformer, dict_saes: Dict[str, Any]):
    model.reset_saes()
    model.reset_hooks(including_permanent=True)
    for _, sae in dict_saes.items():
        sae.reset_hooks()

def attribution(
    model: HookedSAETransformer, 
    dict_saes: Dict[str, Any],
    dataloader: DataLoader,
    feature_set_a: List[int],
    hook_position_a: str,
    hook_position_b: str,
    k: int = 10,
    grad_only: bool = False,
    use_error_term: bool = True,
    direct_effect: bool = True,
) -> Dict[int, List[Dict[str, float]]]:
    
    d_sae_end = dict_saes[hook_position_a].cfg.d_sae
    d_sae_start = dict_saes[hook_position_b].cfg.d_sae
    
    end_node = ConnectionNode(hook_position_a)
    layer = int(hook_position_a.split(".")[1]) # type: ignore
    end_indices = [FeatureIndex([int(feat)], 1) for feat in feature_set_a]
    
    num_end = d_sae_end
    if use_error_term:
        num_end += 1
    
    to_bwd_cache = {}
    bwd_cache = {}
    fwd_cache = {}
    edge_effect = OrderedDict()
    with t.set_grad_enabled(True):
        with _detach_error_term(dict_saes, False, hook_position_a):
            with _setup_forward_model_hook(model, dict_saes, use_error_term):
                with _setup_fwd_bwd_edge_grad_sae_hook(
                    dict_saes,
                    bwd_cache=bwd_cache,
                    to_bwd_cache=to_bwd_cache,
                    fwd_cache=fwd_cache,
                    hook_position_b=hook_position_b,
                    hook_position_a=hook_position_a,
                    direct_effect=direct_effect,
                    use_error_term=use_error_term,
                ):
                    for toks in tqdm(dataloader):
                        clean_token = toks[0].to(model.cfg.device)
                        model.forward(clean_token, return_type=None, stop_at_layer=layer+1)
                        
                        seq = int(clean_token.shape[1])
                        
                        aggregate_dim = [0, 1]
                        to_bwd = cache_to_sparseact(
                            to_bwd_cache, 
                            sae_hook_name(hook_position_a), 
                            error_term_name(hook_position_a) if use_error_term else None,
                        )
                        to_bwd = SparseAct(
                            act = to_bwd.act, resc=(to_bwd.res).sum(dim=-1) if to_bwd.res is not None else None
                        ) # contract error term if exists
                        to_bwd = to_bwd.sum(aggregate_dim).to_tensor()
                        
                        for active_idx, end_index in enumerate(end_indices):
                            if to_bwd[end_index.idx] > 1e-3:
                                to_bwd[end_index.idx].backward(retain_graph=True)
                                rightvec = cache_to_sparseact(
                                    fwd_cache,
                                    sae_hook_name(hook_position_b),
                                    error_term_name(hook_position_b) if use_error_term else None,
                                )
                                if not grad_only:
                                    if edge_effect.get(active_idx) is None:
                                        edge_effect[active_idx] = t.zeros(
                                            (d_sae_start + 1), 
                                            device=clean_token.device,
                                            dtype=t.float64, # to avoid numerical issue
                                        )
                                        
                                    edge_effect[active_idx] += ( # (d_sae+1)
                                        cache_to_sparseact(
                                            bwd_cache, 
                                            sae_hook_name(hook_position_b), 
                                            error_term_name(hook_position_b) if use_error_term else None,
                                        ).to(t.float64) @ rightvec.to(t.float64)
                                    ).sum(aggregate_dim).to_tensor().detach() # (d_sae+1)
                                else:
                                    if edge_effect.get(active_idx) is None:
                                        edge_effect[active_idx] = t.zeros(
                                            (d_sae_start), 
                                            device=clean_token.device,
                                            dtype=t.float64, # to avoid numerical issue
                                        )
                                        
                                    edge_effect[active_idx] += bwd_cache[
                                        sae_hook_name(hook_position_b)
                                    ].to(t.float64).sum(aggregate_dim).detach() # (d_sae)
                        
        model_setup(model, dict_saes)
       
    del to_bwd_cache
    del bwd_cache
    del fwd_cache
    final_assignments = {}
    # active_idx is the enumerate index into end_indices, i.e. the position in feature_set_a, so
    # index by it rather than zipping edge_effect (insertion order = order features first fire,
    # which diverges from feature_set_a once the target set is large enough that some features do
    # not fire in the first batch). Features that never fired above threshold are simply absent.
    for active_idx, feat_idx in enumerate(feature_set_a):
        effect = edge_effect.get(active_idx)
        if effect is None:
            continue
        values, indices = t.topk(effect[:d_sae_start], k=k) # (d_sae_start)

        final_assignments[int(feat_idx)] = [
            {"target_feature": int(f_b), "causal_effect": d} for f_b, d in zip(indices.cpu().numpy(), values.cpu().numpy())
        ]

    return final_assignments