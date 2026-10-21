from tracemalloc import start
from typing import Literal
from .Feature_Graph_Trans import *
from transformer_lens.utils import get_act_name

class Feature_Graph_Cross(Feature_Graph_Trans):
    def __init__(
        self,
        model: HookedSAETransformer,
        saes: Dict[int, List[Tuple[str, Any]]], # {layer: list[{hook_position: HookedSAE}]}, can define granularity here
        use_error_term: bool = False,
    ):
        super().__init__(model, saes, use_error_term)
                
    def process_transcoder(self):
        self.input_hooks = []
        self.output_hooks = []
        self.crosscoders = []
        self.non_crosscoders = []
        check_mlp_out = False
        self.check_attn_out = False
        self.check_resid_pre = False
        for sae in self.dict_saes.values():
            sae = sae.to(self.device)
            if getattr(sae, 'input_hook', False) and getattr(sae, 'output_hooks', False):
                self.input_hooks.append(sae.input_hook)
                self.output_hooks.append(sae.output_hooks[0])
                self.crosscoders.append(sae)
                
                if "mlp_out" in sae.output_hooks[0]:
                    check_mlp_out = True
            else:
                self.non_crosscoders.append(sae)
                
            if "attn_out" in sae.cfg.hook_name:
                self.check_attn_out = True
            if "resid_pre" in sae.cfg.hook_name:
                self.check_resid_pre = True
            
        assert check_mlp_out, "Crosscoder needs to be provided at the mlp_out hook."
             
    @contextmanager
    def _setup_forward_model_hook(self, use_error_term: bool | None = None, transfer_grad: bool = True):
        cross_cache = {}
        # Hook function at transcoder input: caches the activations before transcoder
        def hook_crosscoder_input(activations: Tensor, hook: HookPoint, crosscoder_idx: int):
            cross_cache[crosscoder_idx] = activations.clone()
        
        all_cross_recons = [0.0 for _ in range(len(self.crosscoders))]
        def hook_crosscoder_output(activations: Tensor, hook: HookPoint, crosscoder_idx: int):
            recons, cross_recons = self.crosscoders[crosscoder_idx].forward_crosscoder(
                (cross_cache[crosscoder_idx], activations, all_cross_recons[crosscoder_idx])
            )
            for j in range(crosscoder_idx+1, self.n_layers):
                all_cross_recons[j] += cross_recons[j-crosscoder_idx-1]
            
            if transfer_grad:
                return recons + (activations - activations.detach())
            else:
                return recons
        
        fwd_hooks = []
        for i in range(len(self.crosscoders)):
            fwd_hooks.append((self.input_hooks[i], partial(hook_crosscoder_input, crosscoder_idx=i)))
            fwd_hooks.append((self.output_hooks[i], partial(hook_crosscoder_output, crosscoder_idx=i)))
            
        use_error_term = use_error_term if use_error_term is not None else self.use_error_term
        try: 
            for sae in self._saes_to_list():
                self.model.add_sae(sae, use_error_term)
            for hook, func in fwd_hooks:
                self.model.add_hook(hook, func, dir="fwd")
            
            yield
        finally:
            self.model.reset_saes()
            self.model.reset_hooks() 
            
    def _saes_to_list(self) -> List[Any]:
        return self.non_crosscoders