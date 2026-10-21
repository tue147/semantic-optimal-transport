from tracemalloc import start
from typing import Literal
from .Feature_Graph import *
from transformer_lens.utils import get_act_name

def add_cache(all_cache: Dict[str, float | Tensor], cache: Dict[str, Tensor]):
    for key, value in cache.items():
        if key not in all_cache:
            all_cache[key] = 0
        all_cache[key] += value
        
def _hook_name(sae_name: str, name: str | None):
    return sae_name + "." + str(name)

class Feature_Graph_Trans(Feature_Graph):
    def __init__(
        self,
        model: HookedSAETransformer,
        saes: Dict[int, List[Tuple[str, Any]]], # {layer: list[{hook_position: HookedSAE}]}, can define granularity here
        use_error_term: bool = False,
    ):
        super().__init__(model, saes, use_error_term)
        
        self.process_transcoder()
                
    def process_transcoder(self):
        self.input_hooks = []
        self.output_hooks = []
        self.transcoders = []
        self.non_transcoders = []
        check_mlp_out = False
        self.check_attn_out = False
        self.check_resid_pre = False
        for sae in self.dict_saes.values():
            sae = sae.to(self.device)
            if getattr(sae, 'input_hook', False) and getattr(sae, 'output_hook', False):
                self.input_hooks.append(sae.input_hook)
                self.output_hooks.append(sae.output_hook)
                self.transcoders.append(sae)
                
                if "mlp_out" in sae.output_hook:
                    check_mlp_out = True
            else:
                self.non_transcoders.append(sae)
                
            if "attn_out" in sae.cfg.hook_name:
                self.check_attn_out = True
            if "resid_pre" in sae.cfg.hook_name:
                self.check_resid_pre = True
            
        assert check_mlp_out, "Transcoder needs to be provided at the mlp_out hook."
                
    def forward(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        patch_deleted_comp: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, SparseAct]]:
        '''
        Forward pass of the graph with clean tokens, if the edge exists, replace the activation with corrupted activation
        '''
        self._check_graph()
        self.model_setup()
        
        fwd_cache = {}
        
        with t.no_grad():
            with self._setup_forward_model_hook(transfer_grad=False):
                with self._setup_fwd_sae_hook(
                    fwd_cache=fwd_cache, corrupt_cache=corrupt_cache, patch_deleted_comp=patch_deleted_comp
                ):
                    logits = self.model(clean_token)
        
        cache = self._aggregate_fwd_cache(fwd_cache)
            
        self.model_setup()
        
        return logits, cache
    
    def intervene_components(
        self,
        clean_token: Tensor,
        corrupt_cache: Dict[str, Tensor],
        metric: Callable[[Tensor], Tensor],
        forward_only: bool = False,
        retain_graph: bool = False,
        pass_through_grad: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        self._check_graph()
        self.model_setup()
            
        bwd_cache = {}
            
        with t.set_grad_enabled(True):
            with self._detach_error_term(True):
                with self._setup_forward_model_hook(transfer_grad=kwargs.get("transfer_grad", True)):
                    with self._setup_intervene_grad_sae_hook(
                        bwd_cache=bwd_cache, 
                        corrupt_cache=corrupt_cache,
                        pass_through_grad=pass_through_grad,
                    ):
                        res = metric(self.model(clean_token))
                        if not forward_only:
                            res.backward(retain_graph=retain_graph)
                
        self.model_setup()
        
        return bwd_cache | {"metric": res}
    
    def forward_backward_gradient(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        metric: Callable[[Tensor], Tensor],
        retain_graph: bool = False,
        mode: str = 'node',
        gradient_mode: str = 'standard',
        pass_through_grad: bool = False,
        save_node_grads = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], SparseAct],  # node effects
        Dict[Tuple[Node, Index, Node, Index], Tensor], # edge effects
    ]:
        if mode == 'node':
            if verbose:
                print("Calculating node gradients...")
            if gradient_mode == "standard":
                node_grads, clean_cache = self._gradient_wrt_nodes(
                    clean_token, metric, retain_graph, pass_through_grad, **kwargs
                )
            elif gradient_mode == "graddrop":
                node_grads, clean_cache = self._gradient_wrt_nodes_graddrop(
                    clean_token, metric, pass_through_grad, **kwargs
                )
            elif gradient_mode in ["ig", "ig2", "idg"]:
                if corrupt_cache is None:
                    raise
                node_grads, clean_cache = self._gradient_wrt_nodes_ig(
                    clean_token, corrupt_cache, metric, gradient_mode, retain_graph, verbose, **kwargs
                )
            elif gradient_mode in ["ig_input", "ig2_input", "idg_input"]:
                if corrupt_cache is None:
                    raise
                node_grads, clean_cache = self._gradient_wrt_nodes_ig_input(
                    clean_token, corrupt_cache, metric, gradient_mode, retain_graph, verbose, **kwargs
                )
            elif gradient_mode == "virtual_weight":
                node_grads, clean_cache = self._gradient_wrt_nodes(
                    clean_token, metric, retain_graph, pass_through_grad, **(kwargs | {"vw": True})
                )
            else:
                raise NotImplementedError(f"gradient_mode {gradient_mode} is not supported")
            node_effect = self._attrib_effect_node(node_grads, corrupt_cache, clean_cache)
            
            if save_node_grads:
                self.node_grads = node_grads
                
            return node_effect, {}
        
        elif mode == 'edge':
            node_effect = {}
            if self.node_grads is None:
                # compute node_grads, node_effect to prune out unimportant nodes
                if verbose:
                    print("Calculating node gradients...")
                if gradient_mode == "standard":
                    node_grads, clean_cache = self._gradient_wrt_nodes(
                        clean_token, metric, retain_graph, pass_through_grad, **kwargs
                    )
                elif gradient_mode == "graddrop":
                    node_grads, clean_cache = self._gradient_wrt_nodes_graddrop(
                        clean_token, metric, pass_through_grad, **kwargs
                    )
                elif gradient_mode in ["ig", "ig2", "idg"]:
                    if corrupt_cache is None:
                        raise
                    node_grads, clean_cache = self._gradient_wrt_nodes_ig(
                        clean_token, corrupt_cache, metric, gradient_mode, retain_graph, verbose, **kwargs
                    )
                elif gradient_mode in ["ig_input", "ig2_input", "idg_input"]:
                    if corrupt_cache is None:
                        raise
                    node_grads, clean_cache = self._gradient_wrt_nodes_ig_input(
                        clean_token, corrupt_cache, metric, gradient_mode, retain_graph, verbose, **kwargs
                    )
                elif gradient_mode == "virtual_weight":
                    node_grads, clean_cache = self._gradient_wrt_nodes(
                        clean_token, metric, retain_graph, pass_through_grad, **(kwargs | {"vw": True})
                    )
                else:
                    raise NotImplementedError(f"gradient_mode {gradient_mode} is not supported")
                node_effect = self._attrib_effect_node(node_grads, corrupt_cache, clean_cache)
                
                if save_node_grads:
                    self.node_grads = node_grads
                
                # Pruning
                if kwargs.get('prune', False):
                    if verbose:
                        print("Pruning nodes...")
                    self._prune_nodes(node_effect, verbose, **kwargs)
            
            else:
                if verbose:
                    print("Using saved node gradients...")
                node_grads = self.node_grads
                
            if kwargs.get('gradient_only', False):
                if verbose:
                    print("Returning edge gradients only...")
                node_grads = {}
                for name, sparse_act in self.node_grads.items(): # type: ignore
                    node_grads[name] = sparse_act.to_sparse_like_self(t.ones_like(sparse_act.to_tensor()))
                    del sparse_act
                    
            edge_grads, _ = self._gradient_wrt_edges(
                clean_token, corrupt_cache, node_grads, verbose, **kwargs # type: ignore
            )
            return node_effect, edge_grads
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
    
    def _gradient_wrt_nodes(
        self,
        clean_token: Tensor,
        metric: Callable[[Tensor], Tensor],
        retain_graph: bool = False,
        pass_through_grad: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], SparseAct],  # node effects
        Dict[str, SparseAct]
    ]:
        self._check_graph()
        self.model_setup()
            
        virtual_weight = kwargs.get("vw", False)
        if virtual_weight:
            self._configure_gradient_flow()
            
        fwd_cache = {}
        bwd_cache = {}
            
        with t.set_grad_enabled(True):
            with self._detach_error_term(True):
                with self._setup_forward_model_hook(transfer_grad=kwargs.get("transfer_grad", True) and not virtual_weight):
                    with self._setup_fwd_bwd_grad_sae_hook(
                        fwd_cache=fwd_cache, 
                        bwd_cache=bwd_cache, 
                        pass_through_grad=pass_through_grad,
                        virtual_weight=virtual_weight,
                    ):
                        metric(self.model(clean_token)).backward(retain_graph=retain_graph)
                
        node_grads = self._aggregate_bwd_cache(bwd_cache)
        cache = self._aggregate_fwd_cache(fwd_cache)
            
        self.model_setup()
          
        return node_grads, cache
    
    def _gradient_wrt_nodes_graddrop(
        self,
        clean_token: Tensor,
        metric: Callable[[Tensor], Tensor],
        pass_through_grad: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], SparseAct],  # node effects
        Dict[str, SparseAct]
    ]:
        self._check_graph()
        self.model_setup()
            
        fwd_cache = {}
        bwd_cache = {}
        graddrop_layers = [-1]
        with t.set_grad_enabled(True):
            with self._detach_error_term(True):
                with self._setup_forward_model_hook(transfer_grad=kwargs.get("transfer_grad", True)):
                    with self._setup_fwd_bwd_graddrop_sae_hook(
                        fwd_cache=fwd_cache, 
                        bwd_cache=bwd_cache, 
                        graddrop_layers=graddrop_layers,
                        pass_through_grad=pass_through_grad,
                    ):
                        out = metric(self.model(clean_token))
                        
                        for layer in range(self.model.cfg.n_layers):
                            graddrop_layers[0] = layer
                            out.backward(retain_graph=True)
                            
        # average the gradients                 
        for key in bwd_cache.keys():
            bwd_cache[key] /= (self.model.cfg.n_layers - 1)
            
        node_grads = self._aggregate_bwd_cache(bwd_cache)
        cache = self._aggregate_fwd_cache(fwd_cache)
            
        self.model_setup()
             
        return node_grads, cache
    
    def _gradient_wrt_nodes_ig(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor],
        metric: Callable[[Tensor], Tensor],
        mode: str,
        retain_graph: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], SparseAct],  # node effects
        Dict[str, SparseAct]
    ]:
        steps = kwargs.get("steps", 10)
        
        self._check_graph()
        self.model_setup()

        _, fwd_cache = self.run_model(clean_token)
        bwd_cache = {}
        with t.set_grad_enabled(True):
            with self._detach_error_term(True):
                for target_name in self.dict_saes.keys():
                    for (sae_act, sae_scale), (error_act, error_scale) in IGPathSelector(
                        ig_name=mode, 
                        target_names=[
                            sae_hook_name(target_name), 
                            error_term_name(target_name) if self.use_error_term else None,
                        ],
                        clean_cache=fwd_cache,
                        corrupt_cache=corrupt_cache,
                        clean_token=clean_token,
                        fwd_func=partial(
                            self.intervene_components, 
                            forward_only=True, 
                            retain_graph=False, 
                            verbose=False, 
                            pass_through_grad=True,
                        ),
                        grad_func=partial(
                            self.intervene_components, 
                            forward_only=False, 
                            retain_graph=False, 
                            verbose=False, 
                            pass_through_grad=True,
                        ),
                        metric=metric,
                        n_steps=steps,
                        **kwargs
                    ):
                        with self._setup_forward_model_hook(transfer_grad=kwargs.get("transfer_grad", True)):
                            with self._setup_fwd_bwd_grad_sae_hook_ig( 
                                target_name=target_name,
                                sae_act=sae_act, # type: ignore
                                error_act=error_act,
                                sae_scale=sae_scale, # type: ignore
                                error_scale=error_scale,
                                bwd_cache=bwd_cache,
                            ):
                                metric(self.model(clean_token)).backward(retain_graph=retain_graph)
               
        # average the gradients                 
        for key in bwd_cache.keys():
            bwd_cache[key] /= steps
                
        node_grads = self._aggregate_bwd_cache(bwd_cache)
        cache = self._aggregate_fwd_cache(fwd_cache)
            
        self.model_setup()
          
        return node_grads, cache
    
    def _gradient_wrt_nodes_ig_input(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor],
        metric: Callable[[Tensor], Tensor],
        mode: str,
        retain_graph: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], SparseAct],  # node effects
        Dict[str, SparseAct]
    ]:
        steps = kwargs.get("steps", 10)
        
        self._check_graph()
        self.model_setup()

        corrupt_token = kwargs.get("corrupt_token", None)
        if corrupt_token is None:
            raise 
        embed_clean_cache = {utils.get_act_name("embed"): self.model.embed(clean_token).detach()}
        embed_corrupt_cache = {utils.get_act_name("embed"): self.model.embed(corrupt_token).detach()}

        _, fwd_cache = self.run_model(clean_token)
        bwd_cache = {}
        with t.set_grad_enabled(True):
            with self._detach_error_term(True):
                for path in IGPathSelector(
                    ig_name=mode, 
                    target_names=[utils.get_act_name("embed")],
                    clean_cache=embed_clean_cache,
                    corrupt_cache=embed_corrupt_cache,
                    clean_token=clean_token,
                    fwd_func=partial(
                        self.intervene_components, 
                        forward_only=True, 
                        retain_graph=False, 
                        verbose=False, 
                        pass_through_grad=True,
                    ),
                    grad_func=partial(
                        self.intervene_components, 
                        forward_only=False, 
                        retain_graph=False, 
                        verbose=False, 
                        pass_through_grad=True,
                    ),
                    metric=metric,
                    n_steps=steps,
                    **kwargs
                ):
                    act, scale = path[0]
                    with self._setup_forward_model_hook(transfer_grad=kwargs.get("transfer_grad", True)):
                        with self._setup_fwd_bwd_grad_sae_hook_ig_input( 
                            input_act=act, # type: ignore
                            scale=scale, # type: ignore
                            bwd_cache=bwd_cache,
                        ):
                            metric(self.model(clean_token)).backward(retain_graph=retain_graph)
               
        # average the gradients                 
        for key in bwd_cache.keys():
            bwd_cache[key] /= steps
                
        node_grads = self._aggregate_bwd_cache(bwd_cache)
        cache = self._aggregate_fwd_cache(fwd_cache)
            
        self.model_setup()
             
        return node_grads, cache
    
    def _gradient_wrt_edges(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        node_grads: Dict[Tuple[Node, Index], SparseAct],
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index, Node, Index], Tensor], # edge effects
        Dict[str, SparseAct]
    ]:
        
        self._check_graph()
        self.model_setup()
            
        gradient_mode = kwargs.get('edge_gradient_mode', 'virtual_weight')
        
        # sae cache on clean token
        _, clean_cache = self.forward(clean_token, corrupt_cache=None)
                    
        edge_grads: Dict[Tuple[Node, Index, Node, Index], Tensor] = {}
        for layer, connection in tqdm(self.connection.items(), disable=not verbose):
            if verbose:
                print(f"Layer {layer}:")
            for hook_position_end, list_hook_positions_start in tqdm(connection.items(), disable=not verbose):
                assert hook_position_end in node_grads, f"Node gradient of {hook_position_end} is not provided."
                
                for hook_position_start in list_hook_positions_start:
                    if corrupt_cache is not None:
                        corrupt_sparse_act = cache_to_sparseact(
                            corrupt_cache, 
                            sae_hook_name(hook_position_start[0].name), 
                            error_term_name(hook_position_start[0].name) if self.use_error_term else None,
                        )
                    else:
                        corrupt_sparse_act = 0
                    right_vec = corrupt_sparse_act - clean_cache[hook_position_start[0].name]
                    
                    if gradient_mode == 'virtual_weight':
                        edge_grads[hook_position_start + hook_position_end] = self._edge_attribution(
                            clean_token,
                            hook_position_end,
                            hook_position_start,
                            node_grads[hook_position_end],
                            right_vec,
                            layer,
                            **(kwargs | {"vw": True}),
                        )
                    elif gradient_mode == 'gradient':
                        edge_grads[hook_position_start + hook_position_end] = self._edge_attribution(
                            clean_token,
                            hook_position_end,
                            hook_position_start,
                            node_grads[hook_position_end],
                            right_vec,
                            layer,
                            **kwargs,
                        )
                    else:
                        raise NotImplementedError(f"gradient_mode {gradient_mode} is not supported")
                                  
        return edge_grads, clean_cache
    
    def _edge_attribution(
        self,
        clean_token: Tensor,
        hook_position_end: Tuple[Node, Index],
        hook_position_start: Tuple[Node, Index],
        leftvec: SparseAct,
        rightvec: SparseAct,
        layer: int,
        **kwargs,
    ) -> Tensor:
        d_sae_end = self.dict_saes[hook_position_end[0].name].cfg.d_sae
        d_sae_start = self.dict_saes[hook_position_start[0].name].cfg.d_sae
        
        virtual_weight = kwargs.get("vw", False)
        if virtual_weight:
            self._configure_gradient_flow()
        
        to_bwd_cache = {}
        bwd_cache = {}
        edge_effect = OrderedDict()
        with t.set_grad_enabled(True):
            with self._detach_error_term(False, hook_position_end[0].name):
                with self._setup_forward_model_hook(transfer_grad=False): # must be false, no gradient from intermediate component
                    with self._setup_fwd_bwd_edge_grad_sae_hook(
                        bwd_cache=bwd_cache,
                        to_bwd_cache=to_bwd_cache,
                        hook_position_start=hook_position_start,
                        hook_position_end=hook_position_end,
                    ):
                        self.model.forward(clean_token, return_type=None, stop_at_layer=layer+1)
                        
                        aggregate_dim = [0] if self.token_wise else [0, 1]
                        to_bwd = ( # (seq, d_sae+1) || (d_sae+1)
                            cache_to_sparseact(
                                to_bwd_cache, 
                                sae_hook_name(hook_position_end[0].name), 
                                error_term_name(hook_position_end[0].name) if self.use_error_term else None,
                            ) @ leftvec.detach()
                        ).sum(aggregate_dim).to_tensor()
                        del to_bwd_cache
                        
                        for active_idx, (end_node, end_index) in enumerate(self.active_nodes(*hook_position_end)):
                            if isinstance(end_index, ErrorIndex):
                                # the last index is error: shape (d_sae+1) so last index is d_sae
                                to_bwd[end_index.idx + (d_sae_end,)].backward(retain_graph=True)
                                index = t.tensor(list(end_index.idx + (d_sae_end,)), device=self.device)
                            elif isinstance(end_index, FeatureIndex):
                                to_bwd[end_index.idx].backward(retain_graph=True)
                                index = t.tensor(list(end_index.idx), device=self.device)
                            else:
                                raise ValueError(f"end_index of type {type(end_index)} is not supported.")
                            '''
                            edge_effect shape (seq, d_sae+1, seq, d_sae+1) or (d_sae+1, d_sae+1) in sparse_coo tensor
                            
                            the sparse_coo will have the shape:
                            --> indices of shape (2, num_active) or (1, num_active)
                            --> values of shape (num_active, seq, d_sae+1) or (num_active, d_sae+1)
                            '''
                            edge_effect[active_idx] = (
                                index,
                                ( # (seq, d_sae+1) || (d_sae+1)
                                    cache_to_sparseact(
                                        bwd_cache, 
                                        sae_hook_name(hook_position_start[0].name), 
                                        error_term_name(hook_position_start[0].name) if self.use_error_term else None,
                                    ) @ rightvec
                                ).sum(aggregate_dim).to_tensor()
                            )
                        
                        del bwd_cache
                        
        self.model_setup()
                    
        seq = int(self.seq_length) # type: ignore
        num_end = d_sae_end
        num_start = d_sae_start
        if self.use_error_term:
            num_end += 1
            num_start += 1
        
        if len(edge_effect.keys()) != 0:
            indices = t.stack([val[0] for val in edge_effect.values()], dim=0).T # shape (2, num_active) or (1, num_active)
            values = t.stack([val[1] for val in edge_effect.values()], dim=0) # shape (num_active, seq, d_sae+1) or (num_active, d_sae+1)
        # if no active nodes, return empty tensor
        else:
            indices = t.empty((2, 0) if self.token_wise else (1, 0), dtype=t.long).to(self.device)
            values = t.empty((0, seq, num_start) if self.token_wise else (0, num_start), dtype=t.float).to(self.device)
            
        if self.token_wise:
            return t.sparse_coo_tensor(indices, values, size=(seq, num_end, seq, num_start)).coalesce()
        else:
            return t.sparse_coo_tensor(indices, values, size=(num_end, num_start)).coalesce()
    
    def run_model(
        self, 
        toks: Tensor, 
        use_error_term: bool | None = None,
    ) -> Tuple[Tensor, ActivationCache]:
        """
        Runs an MLP transcoder(s) on a batch of tokens.
        """
        fwd_cache = {}
        with self._setup_forward_model_hook(use_error_term=use_error_term, transfer_grad=False):
            with self._setup_fwd_sae_hook(
                fwd_cache=fwd_cache, corrupt_cache=None, patch_deleted_comp=False, use_error_term=use_error_term
            ):
                logits = self.model(toks)

        return logits, ActivationCache(cache_dict=fwd_cache, model=self.model) # type: ignore
    
    @contextmanager
    def _setup_fwd_sae_hook(
        self, 
        fwd_cache: Dict[str, Tensor],
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None, 
        patch_deleted_comp: bool = False,
        use_error_term: bool | None = None,
    ):    
        use_error_term = use_error_term if use_error_term is not None else self.use_error_term
        
        def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str) -> Tensor:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                if patch_deleted_comp and corrupt_cache is not None:
                    act_mask = self.nodes[(sae_name, (None,))].act == 0 # type: ignore
                    act[:, act_mask] = corrupt_cache[sae_hook_name(sae_name)][:, act_mask]
                fwd_cache[sae_hook_name(sae_name)] = act.detach()
                
            elif (hook.name == "hook_sae_error" or hook.name == error_term_name(sae_name)) and use_error_term:
                if patch_deleted_comp and corrupt_cache is not None:
                    resc_mask = self.nodes[(sae_name, (None,))].resc == 0 # type: ignore
                    act[:, resc_mask] = corrupt_cache[error_term_name(sae_name)][:, resc_mask]
                fwd_cache[error_term_name(sae_name)] = act.detach()
                
            return act
        
        try:
            with self._setup_error_term(use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_fwd, sae_name=sae_name),
                        dir="fwd",
                    )
                yield
        finally:
            for sae in self.dict_saes.values():
                sae.reset_hooks()
                
    @contextmanager
    def _setup_intervene_grad_sae_hook(
        self, 
        bwd_cache: Dict[str, Tensor],
        corrupt_cache: Dict[str, Tensor],
        pass_through_grad: bool,
    ):
        def hook_fwd(act: Tensor, hook: HookPoint) -> Tensor:
            modified_act = act.clone()
            if hook.name in corrupt_cache:
                modified_act.copy_(corrupt_cache[hook.name])
                    
            return modified_act
        
        def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str) -> Tensor:
            modified_act = act.clone()
            if hook.name in corrupt_cache:
                modified_act.copy_(corrupt_cache[hook.name])
            elif f"{sae_name}.{hook.name}" in corrupt_cache:
                modified_act.copy_(corrupt_cache[f"{sae_name}.{hook.name}"])
                    
            return modified_act
        
        def hook_bwd(grad: Tensor, hook: HookPoint) -> None:
            if hook.name in corrupt_cache and hook.name not in bwd_cache:
                bwd_cache[hook.name] = grad.detach()
        
        pass_through_cache = {}
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str) -> None:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                bwd_cache[sae_hook_name(sae_name)] = grad.detach()
                
            elif hook.name == "hook_sae_output" or hook.name == output_hook_name(sae_name):
                if self.use_error_term:
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    bwd_cache[error_term_name(sae_name)] = grad.detach()
                # avoid pass through grad for transcoder
                if pass_through_grad and sae_name not in self.output_hooks:
                    pass_through_cache[output_hook_name(sae_name)] = grad.detach()
                    
            elif hook.name == "hook_sae_input" or hook.name == input_hook_name(sae_name):
                if pass_through_grad:
                    # we have to modify inplace instead of grad = ... and then return grad
                    # because, returning a tensor in bwd pass hook is buggy somehow
                    if sae_name not in self.output_hooks:
                        # no pass through grad for transcoder
                        grad.copy_(pass_through_cache[output_hook_name(sae_name)])
                    else:
                        # avoid gradient flow to upstream from the transcoder
                        grad.zero_() 
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_fwd, sae_name=sae_name),
                        dir="fwd",
                    )
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_bwd, sae_name=sae_name),
                        dir="bwd",
                    )
                self.model.add_hook(lambda name: name in corrupt_cache, hook_fwd, dir="fwd") # type: ignore
                self.model.add_hook(lambda name: name in corrupt_cache, hook_bwd, dir="bwd") # type: ignore
                yield
        finally:
            self.model_setup()
                
    @contextmanager
    def _setup_fwd_bwd_grad_sae_hook(
        self, 
        fwd_cache: Dict[str, Tensor],
        bwd_cache: Dict[str, Tensor],
        pass_through_grad: bool,
        virtual_weight: bool = False,
    ):
        def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str) -> Tensor:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                fwd_cache[sae_hook_name(sae_name)] = act.detach()
                
            elif (hook.name == "hook_sae_error" or error_term_name(sae_name) == hook.name) and self.use_error_term:
                fwd_cache[error_term_name(sae_name)] = act.detach()
                    
            return act
        
        pass_through_cache = {}
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str) -> None:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                bwd_cache[sae_hook_name(sae_name)] = grad.detach()
                
            elif hook.name == "hook_sae_output" or hook.name == output_hook_name(sae_name):
                if self.use_error_term:
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    bwd_cache[error_term_name(sae_name)] = grad.detach()
                # avoid pass through grad for transcoder
                if pass_through_grad and sae_name not in self.output_hooks:
                    pass_through_cache[output_hook_name(sae_name)] = grad.detach()
                    
                # # DEBUG
                # if pass_through_grad and sae_name in self.output_hooks:
                #     pass_through_cache[output_hook_name(sae_name)] = grad.detach()
                    
            elif hook.name == "hook_sae_input" or hook.name == input_hook_name(sae_name):
                if pass_through_grad:
                    # we have to modify inplace instead of grad = ... and then return grad
                    # because, returning a tensor in bwd pass hook is buggy somehow
                    if sae_name not in self.output_hooks:
                        # no pass through grad for transcoder
                        grad.copy_(pass_through_cache[output_hook_name(sae_name)])
                    else:
                        if not virtual_weight:
                            # avoid gradient flow to upstream from the transcoder
                            grad.zero_() 
                            
                        # # DEBUG
                        # else:
                        #     with t.no_grad():
                        #     manual_grad = einops.einsum(
                        #         pass_through_cache[output_hook_name(sae_name)],
                        #         W_dec,
                        #         "b s d_model, d_sae d_model -> b s d_sae"
                        #     ) * (fwd_cache[sae_hook_name(sae_name)] > 0).to(dtype=W_dec.dtype)
                        #     print(sae_hook_name(sae_name))
                        #     manual_grad = einops.einsum(
                        #         manual_grad,
                        #         W_enc,
                        #         "b s d_sae, d_model d_sae -> b s d_model"
                        #     )
                            
                        #     t.testing.assert_close(grad, manual_grad)
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_fwd, sae_name=sae_name),
                        dir="fwd",
                    )
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_bwd, sae_name=sae_name),
                        dir="bwd",
                    )
                yield
        finally:
            for sae in self.dict_saes.values():
                sae.reset_hooks()
            self.model.reset_hooks()
                
    # modified from: https://github.com/safety-research/circuit-tracer/blob/main/circuit_tracer/replacement_model.py
    def _configure_gradient_flow(self):
        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.model.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            self.model.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore

        for param in self.model.parameters():
            param.requires_grad = False

        # def enable_gradient(tensor, hook):
        #     tensor.requires_grad = True
        #     return tensor

        # self.hook_embed.add_hook(enable_gradient, is_permanent=True)
                
    @contextmanager
    def _setup_fwd_bwd_graddrop_sae_hook(
        self, 
        fwd_cache: Dict[str, Tensor],
        bwd_cache: Dict[str, Tensor],
        graddrop_layers: List[int],
        pass_through_grad: bool,
    ):
        def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str) -> Tensor:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                fwd_cache[sae_hook_name(sae_name)] = act.detach()
                
            elif (hook.name == "hook_sae_error" or error_term_name(sae_name) == hook.name) and self.use_error_term:
                fwd_cache[error_term_name(sae_name)] = act.detach()
                    
            return act
        
        pass_through_cache = {}
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str) -> None:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                create_list(bwd_cache, sae_hook_name(sae_name))
                bwd_cache[sae_hook_name(sae_name)] += grad.detach()
                
            elif hook.name == "hook_sae_output" or hook.name == output_hook_name(sae_name):
                if self.use_error_term:
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    create_list(bwd_cache, error_term_name(sae_name))
                    bwd_cache[error_term_name(sae_name)] += grad.detach()
                # avoid pass through grad for transcoder
                if pass_through_grad and sae_name not in self.output_hooks:
                    pass_through_cache[output_hook_name(sae_name)] = grad.detach()
                    
            elif hook.name == "hook_sae_input" or hook.name == input_hook_name(sae_name):
                if pass_through_grad:
                    # we have to modify inplace instead of grad = ... and then return grad
                    # because, returning a tensor in bwd pass hook is buggy somehow
                    if sae_name not in self.output_hooks:
                        # no pass through grad for transcoder
                        grad.copy_(pass_through_cache[output_hook_name(sae_name)])
                    else:
                        # avoid gradient flow to upstream from the transcoder
                        grad.zero_() 
                        
        def hook_graddrop(grad: Tensor, hook: HookPoint, graddrop_layer: List[int] = graddrop_layers):
            if hook.layer() in graddrop_layer:
                grad.zero_() # zero grad for component
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_fwd, sae_name=sae_name),
                        dir="fwd",
                    )
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_bwd, sae_name=sae_name),
                        dir="bwd",
                    )
                self.model.add_hook(
                    lambda name: "attn.hook_z" in name or "mlp.hook_post" in name, hook_graddrop, dir="bwd" # type: ignore
                )
                yield
        finally:
            self.model_setup()
         
    @contextmanager
    def _setup_fwd_bwd_grad_sae_hook_ig(
        self, 
        target_name: str,
        sae_act: Tensor,
        error_act: Tensor | None,
        sae_scale: Tensor,
        error_scale: Tensor | None,
        bwd_cache: Dict[str, Tensor],
    ):  
        def hook_sae_fwd(act: Tensor, hook: HookPoint) -> Tensor:
            modified_act = act.clone()
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(target_name):
                # interpolate for integrated gradients
                modified_act.copy_(sae_act)
                
            elif (hook.name == "hook_sae_error" or error_term_name(target_name) == hook.name) and self.use_error_term:
                # interpolate for integrated gradients
                modified_act.copy_(error_act) # type: ignore
                    
            return modified_act
        
        def hook_sae_bwd(grad: Tensor, hook: HookPoint) -> None:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(target_name):
                create_list(bwd_cache, sae_hook_name(target_name))
                bwd_cache[sae_hook_name(target_name)] += grad.detach() * sae_scale
                
            elif hook.name == "hook_sae_output" or hook.name == output_hook_name(target_name):
                if self.use_error_term:
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    create_list(bwd_cache, error_term_name(target_name))
                    bwd_cache[error_term_name(target_name)] += grad.detach() * error_scale # type: ignore
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    if sae_name != target_name: # only hook the target SAE
                        continue
                    sae.add_hook(
                        lambda name: True,
                        hook_sae_fwd,
                        dir="fwd",
                    )
                    sae.add_hook(
                        lambda name: True,
                        hook_sae_bwd,
                        dir="bwd",
                    )
                yield
        finally:
            for sae in self.dict_saes.values():
                sae.reset_hooks()
                
    @contextmanager
    def _setup_fwd_bwd_grad_sae_hook_ig_input(
        self, 
        input_act: Tensor,
        scale: Tensor,
        bwd_cache: Dict[str, Tensor],
    ):  
        def hook_fwd(act: Tensor, hook: HookPoint) -> Tensor:
            modified_act = act.clone()
            
            # interpolate for integrated gradients
            modified_act.copy_(input_act)
                    
            return modified_act
        
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str) -> None:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                create_list(bwd_cache, sae_hook_name(sae_name))
                bwd_cache[sae_hook_name(sae_name)] += grad.detach() * scale
                
            elif hook.name == "hook_sae_output" or hook.name == output_hook_name(sae_name):
                if self.use_error_term:
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    create_list(bwd_cache, error_term_name(sae_name))
                    bwd_cache[error_term_name(sae_name)] += grad.detach() * scale # type: ignore
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    sae.add_hook(
                        lambda name: True, 
                        partial(hook_sae_bwd, sae_name=sae_name),
                        dir="bwd",
                    )
                self.model.add_hook(
                    utils.get_act_name("embed"), hook_fwd, dir="fwd" # type: ignore
                )
                yield
        finally:
            self.model_setup()
              
    @contextmanager
    def _setup_fwd_bwd_edge_grad_sae_hook(
        self, 
        bwd_cache: Dict[str, Tensor],
        to_bwd_cache: Dict[str, Tensor],
        hook_position_start: Tuple[Node, Index],
        hook_position_end: Tuple[Node, Index],
    ):
        # print(hook_position_end[0].name, hook_position_start[0].name)
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str, bwd_cache: Dict) -> None:
            # print(sae_name, hook.name, t.sum(grad != 0).item())
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                if hook.name == sae_hook_name(hook_position_start[0].name) or _hook_name(sae_name, hook.name) == sae_hook_name(hook_position_start[0].name):
                    # only store the gradient at the start position
                    bwd_cache[sae_hook_name(sae_name)] = grad.detach()
                
            elif hook.name == "hook_sae_output" or hook.name == output_hook_name(sae_name):
                if hook.name == output_hook_name(hook_position_start[0].name) or _hook_name(sae_name, hook.name) == output_hook_name(hook_position_start[0].name):
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    if self.use_error_term:
                        # only store the gradient at the start position
                        bwd_cache[error_term_name(sae_name)] = grad.detach()

            # IMPORTANT NOTE for reproducibility:
            # we zero grad of intermediate components
            # but zero grad the output hook of SAE will ALSO ZERO GRAD the resid mid / post --> no downstream grads
            # so, we instead zero grad of the input hook of SAE 
            elif hook.name == "hook_sae_input" or hook.name == input_hook_name(sae_name):
                if hook.name == input_hook_name(hook_position_end[0].name) or _hook_name(sae_name, hook.name) == input_hook_name(hook_position_end[0].name):
                    pass # no zero grad for end position
                else:
                    grad.zero_()

        def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str, to_bwd_cache: Dict) -> Tensor:
            if hook.name == "hook_sae_acts_post" or hook.name == sae_hook_name(sae_name):
                if  hook.name == sae_hook_name(hook_position_end[0].name) or _hook_name(sae_name, hook.name) == sae_hook_name(hook_position_end[0].name):
                    # store activation for backward later
                    to_bwd_cache[sae_hook_name(sae_name)] = act
                
            elif (hook.name == "hook_sae_error" or error_term_name(sae_name) == hook.name) and self.use_error_term:
                if  hook.name == error_term_name(hook_position_end[0].name) or _hook_name(sae_name, hook.name) == error_term_name(hook_position_end[0].name):
                    # store activation for backward later
                    to_bwd_cache[error_term_name(sae_name)] = act
                    
            return act
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_fwd, sae_name=sae_name, to_bwd_cache=to_bwd_cache),
                        dir="fwd",
                    )
                    sae.add_hook(
                        lambda name: True,
                        partial(hook_sae_bwd, sae_name=sae_name, bwd_cache=bwd_cache),
                        dir="bwd",
                    )
                yield
        finally:
            for sae in self.dict_saes.values():
                sae.reset_hooks()
                
    @contextmanager
    def _setup_forward_model_hook(self, use_error_term: bool | None = None, transfer_grad: bool = True):
        trans_cache = {}
        # Hook function at transcoder input: caches the activations before transcoder
        def hook_transcoder_input(activations: Tensor, hook: HookPoint, transcoder_idx: int):
            trans_cache[transcoder_idx] = activations.clone()

        # Hook function at transcoder output: replaces activations with transcoder output
        def hook_transcoder_output(activations: Tensor, hook: HookPoint, transcoder_idx: int):
            trans_input = trans_cache[transcoder_idx]
            if transfer_grad:
                # allow gradient to pass through the transcoder output and original activations
                return self.transcoders[transcoder_idx]((trans_input, activations)) + (activations - activations.detach())
            else:
                # no original activations gradient, but save memory
                return self.transcoders[transcoder_idx]((trans_input, activations))
        
        fwd_hooks = []
        for i in range(len(self.transcoders)):
            fwd_hooks.append((self.input_hooks[i], partial(hook_transcoder_input, transcoder_idx=i)))
            fwd_hooks.append((self.output_hooks[i], partial(hook_transcoder_output, transcoder_idx=i)))
            
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
    
    # @contextmanager  
    # def _setup_error_term(self, use_error_term: bool | None = None):
    #     if use_error_term is None:
    #         use_error_term = self.use_error_term
        
    #     orig_use_error_term = {}
    #     try: 
    #         for sae_name, sae in self.dict_saes.items():
    #             sae.use_error_term = use_error_term
    #             orig_use_error_term[sae_name] = sae.use_error_term
    #         yield
    #     finally:
    #         for sae_name, sae in self.dict_saes.items():
    #             sae.use_error_term = orig_use_error_term[sae_name]   
            
    def _saes_to_list(self) -> List[Any]:
        return self.non_transcoders