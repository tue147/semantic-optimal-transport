from graph.Feature_Graph import *
import random as rd
from sae.SAE_Tools import *
from sae.autoencoder.SAE_Wrapper import HookedSAE
from test_matching.util import *

def _hook_name(sae_name: str, name: str | None):
    return sae_name + "." + str(name)

def compute_variance_explained(target, prediction):
    """
    Computes R^2 (variance explained) for two tensors.
    """
    # 1. Calculate Sum of Squares of Residuals
    # (y - y_hat)^2
    n_dim = len(target.shape)
    ss_res = t.sum((target - prediction) ** 2, dim=tuple(range(1, n_dim))).mean()
    
    # 2. Calculate Total Sum of Squares
    # (y - y_mean)^2
    target_mean = t.mean(target)
    ss_tot = t.sum((target - target_mean) ** 2, dim=tuple(range(1, n_dim))).mean()
    
    # 3. Compute R-squared
    # Handle edge case where ss_tot is 0 to avoid division by zero
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot).item()

@contextmanager
def setup_layer_skipping_sae_hook(
    sae_a: HookedSAE,
    sae_b: HookedSAE,
    end_node: List[Node],
    end_index: List[Index],
    start_node: List[Node],
    start_index: List[Index],
    var_ex_cache: List[float],
    use_error_term: bool,
):
    cache = {}
    def cache_fn(tens: Tensor, hook: HookPoint, sae_b: Any, cache: Dict[Tuple[Node, Index], Tensor]) -> None:
        if (
            hook.name == sae_hook_name(sae_b.cfg.hook_name) or 
            _hook_name(sae_b.cfg.hook_name, hook.name) == sae_hook_name(sae_b.cfg.hook_name)
        ):
            for node, index in zip(start_node, start_index):
                cache[(node, index)] = tens[index.as_index].detach()
                
    def patch_fn(tens: Tensor, hook: HookPoint, sae_a: Any, cache: Dict[Tuple[Node, Index], Tensor]) -> Tensor:
        if (
            hook.name == sae_hook_name(sae_a.cfg.hook_name) or 
            _hook_name(sae_a.cfg.hook_name, hook.name) == sae_hook_name(sae_a.cfg.hook_name)
        ):
            for i, (node, index) in enumerate(zip(end_node, end_index)):
                tens[index.as_index] = cache[(start_node[i], start_index[i])]
        elif (
            hook.name == input_hook_name(sae_a.cfg.hook_name) or 
            _hook_name(sae_a.cfg.hook_name, hook.name) == input_hook_name(sae_a.cfg.hook_name)
        ):
            cache[input_hook_name(sae_a.cfg.hook_name)] = tens.detach() # type: ignore
        elif (
            hook.name == recons_hook_name(sae_a.cfg.hook_name) or  # we check reconstruction only
            _hook_name(sae_a.cfg.hook_name, hook.name) == recons_hook_name(sae_a.cfg.hook_name)
        ):
            var_ex_cache.append(compute_variance_explained(cache[input_hook_name(sae_a.cfg.hook_name)], tens.detach())) # type: ignore
                
        return tens
    
    orig_use_error_term = {}
    try:
        orig_use_error_term[sae_a.cfg.hook_name] = sae_a.use_error_term
        sae_a.use_error_term = use_error_term
        sae_a.add_hook(
            lambda name: True,
            partial(patch_fn, sae_a=sae_a, cache=cache), # type: ignore
            dir="fwd",
        )
        
        orig_use_error_term[sae_b.cfg.hook_name] = sae_b.use_error_term
        sae_b.use_error_term = use_error_term
        sae_b.add_hook(
            lambda name: True,
            partial(cache_fn, sae_b=sae_b, cache=cache), # type: ignore
            dir="fwd",
        )
        yield
    finally:
        sae_a.reset_hooks()
        sae_a.use_error_term = orig_use_error_term[sae_a.cfg.hook_name]
        
        sae_b.reset_hooks()
        sae_b.use_error_term = orig_use_error_term[sae_b.cfg.hook_name]

@t.no_grad()
def get_activation_layer_skipping(
    model: HookedSAETransformer,
    sae_a: HookedSAE,
    sae_b: HookedSAE,
    end_node: List[Node],
    end_index: List[Index],
    start_node: List[Node],
    start_index: List[Index],
    data_toks: Tensor,
    use_error_term: bool,
    device: t.device | str,
    dtype: t.dtype = t.float32,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tuple[float, float]:  
    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    assert len(end_node) == len(end_index)
    assert len(start_node) == len(start_index)
    assert len(end_node) == len(start_node)
    all_loss = []
    all_var_ex = []
    original_dtype = model.cfg.dtype
    model = model.to(dtype, print_details=False) # type: ignore
    sae_a = sae_a.to(dtype)
    sae_b = sae_b.to(dtype)
    for token in tqdm(dataloader, disable=not verbose):
        with setup_layer_skipping_sae_hook(
            sae_a=sae_a,
            sae_b=sae_b,
            end_node=end_node,
            end_index=end_index,
            start_node=start_node,
            start_index=start_index,
            var_ex_cache=all_var_ex,
            use_error_term=use_error_term, 
        ):
            with setup_forward_model_hook(model, [sae_a, sae_b], use_error_term):
                logits = model(token[0])
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = token[0][..., 1:].contiguous()
                
                loss_fct = t.nn.CrossEntropyLoss(reduction='none')
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1).to(device)
                )
        
                all_loss.append(loss)
                
    model = model.to(original_dtype, print_details=False) # type: ignore
    sae_a = sae_a.to(original_dtype)
    sae_b = sae_b.to(original_dtype)
    
    return t.cat(all_loss).mean().item(), np.mean(all_var_ex) # type: ignore