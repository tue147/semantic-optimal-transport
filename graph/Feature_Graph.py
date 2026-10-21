import torch as t
from torch import Tensor
from .Graph_Template import Graph, GraphName, Node, Index
from typing import List, Tuple, Dict, Union, Callable, Any
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    ActivationCache,
)
from copy import deepcopy
from collections import OrderedDict, defaultdict
from warnings import warn
from contextlib import contextmanager
from sae_lens import (
    HookedSAETransformer,
)
from functools import partial
from itertools import product
from collections import OrderedDict
from .Graph_utils import nested_dict_to_string
from .Sparse_act import SparseAct
from torch import sparse_coo_tensor
import einops

def cache_to_sparseact(
    cache: Dict[str, Tensor] | ActivationCache, 
    act_name: str, 
    res_name: str | None = None, 
    resc_name: str | None = None,
) -> SparseAct:
    return SparseAct(
        act=cache[act_name],
        res=cache[res_name] if res_name is not None else None,
        resc=cache[resc_name] if resc_name is not None else None,
    )
    
class ConnectionNode(Node):
    def __init__(self, name: str):
        self._name = name
        
    @property
    def name(self) -> str:
        return self._name
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ConnectionNode):
            return self._name == other.name
        elif isinstance(other, str):
            return self._name == other
        else:
            raise NotImplementedError("other is not an instance of ConnectionNode or str")
    
    def __repr__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
class ConnectionIndex(Index):
    def __init__(
        self,
        list_index: tuple[int|None, ...] | None = None,
    ):
        if list_index is None:
            self.list_index = (None,)
        else:
            for index in list_index:
                assert type(index) == int or index == None, "index is not an instance of int or None"
            self.list_index = list_index
        
    @property
    def as_index(self) -> Tuple[int | slice, ...]:
        return tuple(slice(None) if x is None else x for x in self.list_index) # for indexing
    
    def __repr__(self) -> str:
        ret = "["
        for idx, x in enumerate(self.list_index):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += ":"
            elif type(x) == int:
                ret += str(x)
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ConnectionIndex):
            return self.list_index == other.list_index
        elif isinstance(other, tuple):
            return self.list_index == other
        else:
            raise NotImplementedError("other is not an instance of ConnectionIndex or tuple")
    
    def __hash__(self) -> int:
        return hash(self.list_index)
    
class FeatureIndex(ConnectionIndex):
    def __init__(
        self, 
        idx: Tuple[int, ...] | List[int], 
        length = 2,
    ):
        self.idx = tuple(idx)
        super().__init__(return_idx(idx, length))
        
class FeatureErrorIndex(ConnectionIndex):
    def __init__(
        self, 
        idx: Tuple[int, ...] | List[int], 
        length = 2,
    ):
        self.idx = tuple(idx)
        super().__init__(return_idx(idx, length))
    
class ErrorIndex(ConnectionIndex):
    def __init__(
        self, 
        idx: Tuple[int, ...] | List[int],
        length = 1,
    ):
        self.idx = tuple(idx)
        super().__init__(return_idx(idx, length))

def sae_hook_name(sae_name: str) -> str:
    return f'{sae_name}.hook_sae_acts_post' # e.g. ...hook_resid_pre.hook_sae_acts_post

def error_term_name(sae_name: str) -> str:
    return f'{sae_name}.hook_sae_error'

def output_hook_name(sae_name: str) -> str:
    return f'{sae_name}.hook_sae_output'

def input_hook_name(sae_name: str) -> str:
    return f'{sae_name}.hook_sae_input'

def recons_hook_name(sae_name: str) -> str:
    return f'{sae_name}.hook_sae_recons'

def return_idx(idx: Tuple | List, length: int = 2) -> Tuple:
    assert len(idx) <= length, "The length of idx must be less than or equal to length"
    return tuple([None for _ in range(length - len(idx))] + list(idx))
    

class Feature_Graph(Graph):
    def __init__(
        self,
        model: HookedSAETransformer,
        saes: Dict[int, List[Tuple[str, Any]]], # {layer: list[{hook_position: HookedSAE}]}, can define granularity here
        use_error_term: bool = False,
    ):
        '''
        '''
        self.model = model
        self.cfg = model.cfg
        self.use_error_term = use_error_term
        self.device = self.cfg.device
        
        self.n_layers = self.cfg.n_layers
        # self.n_heads = self.cfg.n_heads
        
        assert len(saes) == self.n_layers, "please provide SAEs"
        self.saes = saes
        self.dict_saes: Dict[str, Any] = self.extract_saes(saes)
        
        self.reset_graph()
        
        self.node_grads = None # node grads buffer
        
    def graph_type(self) -> str:
        return GraphName.feature_graph
        
    def extract_saes(self, saes: Dict[int, List[Tuple[str, Any]]]) -> Dict[str, Any]:
        dict_saes = {}
        for layer in range(self.n_layers):
            for hook_position, sae in saes[layer]:
                dict_saes[hook_position] = sae
        return dict_saes
        
    def build_default_connection(
        self,
        seq_length: int,
        token_wise: bool = False,
        inter: bool = True,
        intra: bool = False,
    ) -> Dict:
        '''
        Sample tokens to get the shape of the activations, necessary to build the graph of features
        If token_wise, the each node is a feature of a token, else, the graph is built feature-wise
        '''
        self.reset_graph()
        for layer in range(self.n_layers):
            # Setup
            self.connection[layer] = OrderedDict()
            for hook_position, _ in self.saes[layer]:
                self.connection[layer][(ConnectionNode(hook_position), ConnectionIndex())] = []
                
            # Inter-layer connections
            if inter:
                for i, (hook_position_end, _) in reversed(list(enumerate(self.saes[layer]))):
                    for j, (hook_position_start, _) in enumerate(self.saes[layer]):
                        if i > j:
                            self.connection[layer][(ConnectionNode(hook_position_end), ConnectionIndex())].append(
                                (ConnectionNode(hook_position_start), ConnectionIndex())
                            )
                            
            # Intra-layer connections
            if intra:
                for prev_layer in range(layer):
                    for hook_position_end, _ in self.saes[layer]:
                        for hook_position_start, _ in self.saes[prev_layer]:
                            self.connection[layer][(ConnectionNode(hook_position_end), ConnectionIndex())].append(
                                (ConnectionNode(hook_position_start), ConnectionIndex())
                            )
            # Sequential connection layer-1 --> layer
            else:
                if layer == 0:
                    continue
                for hook_position_start, _ in self.saes[layer-1]:
                    # connect to the first node of the next layer
                    hook_position_end = self.saes[layer][0][0] 
                    self.connection[layer][(ConnectionNode(hook_position_end), ConnectionIndex())].append(
                        (ConnectionNode(hook_position_start), ConnectionIndex())
                    )
                
        self._build_nodes(seq_length, token_wise)
        self._build_graphs()
        return self.connection
    
    def _build_nodes(self, seq_length: int, token_wise: bool) -> None:
        self.seq_length = seq_length
        self.token_wise = token_wise
        
        for sae_name, sae in self.dict_saes.items():
            node_shape = (self.seq_length, sae.cfg.d_sae)
            self.nodes[(ConnectionNode(sae_name), ConnectionIndex())] = SparseAct(
                t.ones(node_shape, requires_grad=False).to(self.device),
                None,
                t.ones(self.seq_length, requires_grad=False).to(self.device) if self.use_error_term else None,
            )
            self.node_scores[(ConnectionNode(sae_name), ConnectionIndex())] = SparseAct(
                t.zeros(node_shape, requires_grad=False).to(self.device),
                None,
                t.zeros(self.seq_length, requires_grad=False).to(self.device) if self.use_error_term else None,
            )
            
    def _build_graphs(self) -> None:
        assert self.nodes, "Please build the nodes first"
        for _, connections in self.connection.items():
            for hook_position_end, list_hook_positions_start in connections.items():
                self.edges[hook_position_end] = OrderedDict()
                self.edge_scores[hook_position_end] = OrderedDict()
                
                for hook_position_start in list_hook_positions_start:
                    self.edges[hook_position_end][hook_position_start] = t.zeros([0]) # place holder
                    self.edge_scores[hook_position_end][hook_position_start] = t.zeros([0]) # place holder
        
    def reset_graph(self) -> None:
        self.connection: OrderedDict[int, OrderedDict[Tuple[Node, Index], List[Tuple[Node, Index]]]] = OrderedDict() # {layer: {hook_position_end: [hook_position_start]}}
        self.nodes: Dict[Tuple[Node, Index], SparseAct] = {} # {hook_position: SparseAct[act: Tensor['seq d_sae'], resc: Tensor['seq']]} 
        self.node_scores: Dict[Tuple[Node, Index], SparseAct] = {} 
        self.edges: OrderedDict[Tuple[Node, Index], OrderedDict[Tuple[Node, Index], Tensor]] = OrderedDict() # {hook_position_end: {hook_position_start: Tensor}}
        self.edge_scores: OrderedDict[Tuple[Node, Index], OrderedDict[Tuple[Node, Index], Tensor]] = OrderedDict()
        self.seq_length: int | None = None
        self.token_wise: bool | None = None
        
    def _check_graph(self) -> None:
        assert len(self.connection) > 0, "Graph is empty"
        assert len(self.nodes) > 0, "Nodes is empty"
        assert len(self.node_scores) > 0, "Node scores is empty"
        assert len(self.edges) > 0, "Edges is empty"
        assert len(self.edge_scores) > 0, "Edge scores is empty"
        assert self.seq_length is not None
        assert self.token_wise is not None
        
    def __repr__(self) -> str:
        self._check_graph()
        return nested_dict_to_string(self.connection, indent=4)
    
    def _active_nodes(
        self,
        sae_name: Node,
        node_idx: Index,
        reverse: bool = False,
    ) -> Tuple[List, ...]:
        self._check_graph()
        if self.token_wise:
            node = self.nodes[(sae_name, node_idx)] # act: (seq, d_sae), resc: (seq) 
        else:
            node = self.nodes[(sae_name, node_idx)].mean(dim=0) # act: (d_sae), resc: () - COLAPSED TENSOR
        
        if reverse:
            active_node = (node.act == 0).nonzero().tolist()
            active_error = (node.resc == 0).nonzero().tolist() if self.use_error_term else [] # type: ignore
        else:
            active_node = node.act.nonzero().tolist()
            active_error = node.resc.nonzero().tolist() if self.use_error_term else [] # type: ignore
            
        return active_node, active_error
            
    def active_nodes(
        self,
        sae_name: Node,
        node_idx: Index,
        reverse: bool = False,
    ) -> List[Tuple[Node, Index]]:
        active_node, active_error = self._active_nodes(sae_name, node_idx, reverse)
        
        feat_list = [
            (sae_name, FeatureIndex(idx)) 
            for idx in active_node
        ]
        error_list = [
            (sae_name, ErrorIndex(idx))
            for idx in active_error
        ]
        return feat_list + error_list # type: ignore
        
    def add_node(
        self, 
        sae_name: Node, 
        node_idx: Index,
    ) -> None:
        self._check_graph()
        pass # no implementation needed
        
    def add_edge(
        self, 
        sae_name_start: Node, 
        node_idx_start: Index, 
        sae_name_end: Node, 
        node_idx_end: Index,
    ) -> None:
        self._check_graph()
        pass # no implementation needed
    
    def delete_node(
        self, 
        sae_name: Node,
        node_idx: Index, 
    ) -> None:
        self._check_graph()
        pass # no implementation needed
    
    def delete_edge(
        self, 
        sae_name_start: Node, 
        node_idx_start: Index, 
        sae_name_end: Node, 
        node_idx_end: Index,
    ) -> None:
        self._check_graph()
        pass # no implementation needed
    
    def _check_node_shape(self, sae_name: Node, sparseact: SparseAct):
        try:
            if self.token_wise:
                assert sparseact.act.shape == t.Size([self.seq_length, self.dict_saes[sae_name.name].cfg.d_sae]) # type: ignore
            else:
                assert sparseact.act.shape == t.Size([self.dict_saes[sae_name.name].cfg.d_sae])
            if self.use_error_term:
                if self.token_wise:
                    assert sparseact.resc.shape == t.Size([self.seq_length]) # type: ignore
                else:
                    assert sparseact.resc.numel() == 1 # type: ignore
        except:
            raise ValueError("Wrong input shape for nodes.")     
    
    def update_node(
        self, 
        sae_name: Node,
        node_idx: Index,  
        value_and_mask: Tuple[SparseAct, SparseAct],
    ) -> None:
        self._check_graph()
        value, mask = value_and_mask
        assert isinstance(value, SparseAct), "value is not an instance of SparseAct"
        assert isinstance(mask, SparseAct), "mask is not an instance of SparseAct"
        self._check_node_shape(sae_name, value)
        
        if not self.token_wise:
            # convert to the right format of (seq, d_sae) and (seq) 
            value_act = einops.repeat(value.act, 'd_sae -> seq d_sae', seq=self.seq_length)
            mask_act = einops.repeat(mask.act, 'd_sae -> seq d_sae', seq=self.seq_length)
            value_resc = einops.repeat(value.resc, ' -> seq', seq=self.seq_length) if self.use_error_term else None
            mask_resc = einops.repeat(mask.resc, ' -> seq', seq=self.seq_length) if self.use_error_term else None
            
            value = SparseAct(value_act, None, value_resc)
            mask = SparseAct(mask_act, None, mask_resc)
            
        self.nodes[(sae_name, node_idx)] = mask.to(t.float32)
        self.node_scores[(sae_name, node_idx)] = value
    
    def update_edge(
        self, 
        sae_name_start: Node, 
        node_idx_start: Index, 
        sae_name_end: Node, 
        node_idx_end: Index,
        value_and_mask: Tuple[Tensor, Tensor],
    ) -> None:
        self._check_graph()
        value, mask = value_and_mask
        assert isinstance(value, Tensor), "value is not an instance of Tensor"
        assert isinstance(mask, Tensor), "mask is not an instance of Tensor"
        self.edges[(sae_name_end, node_idx_end)][(sae_name_start, node_idx_start)] = mask
        self.edge_scores[(sae_name_end, node_idx_end)][(sae_name_start, node_idx_start)] = value
    
    def find_deleted_nodes(self, reverse = False) -> List[Tuple[Node, Index]]:
        self._check_graph()
        
        all_list = []
        for sae_name, node_idx in self.nodes.keys():
            inactive_node, inactive_error = self._active_nodes(sae_name, node_idx, not reverse)
            feat_list = [
                (sae_name, FeatureIndex(idx))
                for idx in inactive_node
            ]
            error_list = [
                (sae_name, ErrorIndex(idx))
                for idx in inactive_error
            ]
            all_list += feat_list + error_list
        
        return all_list
    
    def find_deleted_edges(self, reverse = False) -> List[Tuple[Node, Index, Node, Index]]:
        self._check_graph()
        
        all_list = []
        for end_node, end_idx in self.edges.keys():
            for start_node, start_idx in self.edges[(end_node, end_idx)].keys():
                edge = self.edges[(end_node, end_idx)][(start_node, start_idx)]
                if not reverse:
                    inactive_node = (edge.values() == 0) # (num_active, seq, d_sae+1) or (num_active, d_sae+1)
                else:
                    inactive_node = edge.values()
                    
                d_sae_end = self.dict_saes[end_node.name].cfg.d_sae
                d_sae_start = self.dict_saes[start_node.name].cfg.d_sae
                    
                active_idx = [index for _, index in self.active_nodes(start_node, start_idx)]
                revised_active_nodes = [
                    index.idx if isinstance(index, FeatureIndex) else index.idx + (d_sae_start,) # type: ignore
                    for index in active_idx
                ]
                
                for num_active_idx in range(edge.indices().shape[1]):
                    end_node_idx = edge.indices()[:, num_active_idx].tolist()
                    # check if the end_node_idx is the error term
                    check_error = True if self.use_error_term and end_node_idx[-1] == d_sae_end else False
                    # create the end_index class to be used in the tuple
                    end_index = FeatureIndex(end_node_idx) if not check_error else ErrorIndex(end_node_idx[:-1])
                    
                    for i, idx in enumerate(revised_active_nodes):
                        if inactive_node[(num_active_idx,) + idx] > 0:
                            all_list.append(
                                (
                                    start_node, 
                                    active_idx[i], 
                                    end_node, 
                                    end_index,
                                )
                            )
        
        return all_list
    
    def iterate_nodes(self) -> List[Tuple[Node, Index]]:
        self._check_graph()
        return list(self.nodes.keys())
    
    def iterate_edges(self) -> List[Tuple[Node, Index, Node, Index]]:
        self._check_graph()
        all_edges = []
        for end, edges in self.edges.items():
            for start, edge in edges.items():
                all_edges.append(start + end)
        return all_edges
    
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
            with self._setup_forward_model_hook():
                with self._setup_fwd_sae_hook(
                    fwd_cache=fwd_cache, corrupt_cache=corrupt_cache, patch_deleted_comp=patch_deleted_comp
                ):
                    logits = self.model(clean_token)

        cache = self._aggregate_fwd_cache(fwd_cache)
            
        self.model_setup()
    
        return logits, cache
    
    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, SparseAct]]:
        return self.forward(*args, **kwargs)
    
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
        
    def _attrib_effect_node(
        self,
        grads: Dict,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        clean_cache: Dict[str, SparseAct],
    ) -> Dict:
        attrib_effect = {}
        aggregate_dim = [0] if self.token_wise else [0, 1]
        
        for (node, index), grad in grads.items():
            if corrupt_cache is not None:
                corrupt_sparse_act = cache_to_sparseact(
                    corrupt_cache, 
                    sae_hook_name(node.name), 
                    error_term_name(node.name) if self.use_error_term else None,
                )
                corrupt_sparse_act.act = corrupt_sparse_act.act * (clean_cache[node.name].act > 0).to(self.model.cfg.dtype) # only consider the active nodes in the graph
            else:
                corrupt_sparse_act = 0
            attrib_effect[(node, index)] = ( # act: (seq, d_sae), resc: (seq) || act: (d_sae), resc: ()
                grad @
                (corrupt_sparse_act - clean_cache[node.name])
            ).sum(aggregate_dim)
            
        return attrib_effect
    
    def _aggregate_bwd_cache(self, bwd_cache: Dict[str, Tensor] | ActivationCache) -> Dict[Tuple[Node, Index], SparseAct]:
        grads = {}
        for node, index in self.nodes.keys():
            grads[(node, index)] = cache_to_sparseact(
                bwd_cache,
                sae_hook_name(node.name),
                error_term_name(node.name) if self.use_error_term else None,
            )
        return grads
    
    def _aggregate_fwd_cache(self, fwd_cache: Dict[str, Tensor] | ActivationCache) -> Dict[str, SparseAct]:
        cache = {}
        for sae_name in self.dict_saes.keys():
            cache[sae_name] = cache_to_sparseact(
                fwd_cache,
                sae_hook_name(sae_name),
                error_term_name(sae_name) if self.use_error_term else None,
            )
        return cache
    
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
        '''
        Forward pass of the graph with clean tokens, if the edge exists, replace the activation with corrupted activation
        Backward pass on the graph wrt the metric
        Return the gradients wrt nodes, edges, and activation cache
        '''
        self._check_graph()
        self.model_setup()

        bwd_cache = {}
        fwd_cache = {}
        
        with t.set_grad_enabled(True):
            with self._detach_error_term(True):
                with self._setup_forward_model_hook():
                    with self._setup_fwd_bwd_grad_sae_hook(
                        fwd_cache=fwd_cache, 
                        bwd_cache=bwd_cache, 
                        pass_through_grad=pass_through_grad,
                    ):
                        metric(self.model(clean_token)).backward(retain_graph=retain_graph)
                
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
            
        gradient_mode = kwargs.get('edge_gradient_mode', 'gradient')
            
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
                    
                    if gradient_mode == "gradient":
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
        
        to_bwd_cache = {}
        bwd_cache = {}
        edge_effect = OrderedDict()
        with t.set_grad_enabled(True):
            with self._detach_error_term(False, hook_position_end[0].name):
                with self._setup_forward_model_hook():
                    with self._setup_fwd_bwd_edge_grad_sae_hook(
                        bwd_cache=bwd_cache,
                        to_bwd_cache=to_bwd_cache,
                        hook_position_start=hook_position_start,
                        hook_position_end=hook_position_end,
                        direct_effect=kwargs.get('direct_effect', True)
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

    def _prune_nodes(
        self,
        node_effect: Dict[Tuple[Node, Index], SparseAct],
        verbose: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("node_threshold", None) is None:
            raise ValueError("Please provide the node_threshold")
        else:
            node_threshold = kwargs.get("node_threshold")
            assert isinstance(node_threshold, (float, int, Tensor)), "node_threshold must be a int, float, or Tensor"
            
            for node, index in tqdm(node_effect.keys(), disable=not verbose):
                if kwargs.get("reverse_prune_node", False):
                    effect_feat_mask = node_effect[(node, index)].act.abs() < node_threshold
                else:
                    effect_feat_mask = node_effect[(node, index)].act.abs() > node_threshold
                    
                if verbose:
                    num_pruned = (~effect_feat_mask).sum().item()
                    print(f"{(node, index)}: pruned ({num_pruned / effect_feat_mask.numel() * 100:.5f}%) features")
                
                if self.use_error_term:
                    if kwargs.get("reverse_prune_node", False):
                        effect_error_mask = node_effect[(node, index)].resc.abs() < node_threshold # type: ignore
                    else:
                        effect_error_mask = node_effect[(node, index)].resc.abs() > node_threshold # type: ignore
                        
                    if verbose:
                        num_pruned = (~effect_error_mask).sum().item()
                        print(f"{(node, index)}: pruned ({num_pruned / effect_error_mask.numel() * 100:.5f}%) errors")
                        
                # replace the nodes with mask to prune
                mask = SparseAct(
                    act=effect_feat_mask,
                    resc=effect_error_mask if self.use_error_term else None, # type: ignore
                )
                self.nodes[(node, index)] = self.nodes[(node, index)] * mask
            
    def model_setup(self):
        self.model.reset_saes()
        self.model.reset_hooks(including_permanent=True)
        for _, sae in self.dict_saes.items():
            sae.reset_hooks()
    
    def run_model(
        self, 
        toks: Tensor, 
        use_error_term: bool | None = None,
    ) -> Tuple[Tensor, ActivationCache]:
        """
        Runs an MLP transcoder(s) on a batch of tokens.
        """
        fwd_cache = {}
        with self._setup_forward_model_hook(use_error_term=use_error_term):
            with self._setup_fwd_sae_hook(
                fwd_cache=fwd_cache, corrupt_cache=None, patch_deleted_comp=False, use_error_term=use_error_term
            ):
                logits = self.model(toks)

        return logits, ActivationCache(cache_dict=fwd_cache, model=self.model) # type: ignore
    
    def _saes_to_list(self) -> List[Any]:
        return [sae for _, sae in self.dict_saes.items()]

    @contextmanager
    def _setup_fwd_bwd_grad_sae_hook(
        self, 
        fwd_cache: Dict[str, Tensor],
        bwd_cache: Dict[str, Tensor],
        pass_through_grad: bool,
    ):
        pass_through_cache = {}
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str) -> None:
            if hook.name == sae_hook_name(sae_name):
                bwd_cache[sae_hook_name(sae_name)] = grad.detach()
                
            elif hook.name == output_hook_name(sae_name):
                if self.use_error_term:
                    # we have: output = recon + stop_grad(error_term)
                    # so, the TRUE error_grad (if not stop_grad) is output_grad 
                    # due to: output = recon + error_term
                    bwd_cache[error_term_name(sae_name)] = grad.detach()
                if pass_through_grad:
                    pass_through_cache[output_hook_name(sae_name)] = grad.detach()
                    
            elif hook.name == input_hook_name(sae_name):
                if pass_through_grad:
                    # we have to modify inplace instead of grad = ... and then return grad
                    # because, returning a tensor in bwd pass hook is buggy somehow
                    grad.copy_(pass_through_cache[output_hook_name(sae_name)])
                    
        def hook_sae_fwd(act: Tensor, hook: HookPoint, sae_name: str) -> Tensor:
            if hook.name == sae_hook_name(sae_name):
                fwd_cache[sae_hook_name(sae_name)] = act.detach()
                
            elif error_term_name(sae_name) == hook.name and self.use_error_term:
                fwd_cache[error_term_name(sae_name)] = act.detach()
                    
            return act
        
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
            
    @contextmanager
    def _setup_fwd_bwd_edge_grad_sae_hook(
        self, 
        bwd_cache: Dict[str, Tensor],
        to_bwd_cache: Dict[str, Tensor],
        hook_position_start: Tuple[Node, Index],
        hook_position_end: Tuple[Node, Index],
        direct_effect: bool = True,
    ):
        def hook_sae_bwd(grad: Tensor, hook: HookPoint, sae_name: str, bwd_cache: Dict) -> None:
            if hook.name == sae_hook_name(hook_position_start[0].name):
                # only store the gradient at the start position
                bwd_cache[sae_hook_name(hook_position_start[0].name)] = grad.detach()
                
            elif hook.name == output_hook_name(hook_position_start[0].name):
                # we have: output = recon + stop_grad(error_term)
                # so, the TRUE error_grad (if not stop_grad) is output_grad 
                # due to: output = recon + error_term
                if self.use_error_term:
                    # only store the gradient at the start position
                    bwd_cache[error_term_name(hook_position_start[0].name)] = grad.detach()

            # IMPORTANT NOTE for reproducibility:
            # we zero grad of intermediate components
            # but zero grad the output hook of SAE will ALSO ZERO GRAD the resid mid / post --> no downstream grads
            # so, we instead zero grad of the input hook of SAE 
            elif hook.name == input_hook_name(sae_name):
                if hook.name != input_hook_name(hook_position_end[0].name) and direct_effect:
                    grad.zero_()

        def hook_sae_fwd(act: Tensor, hook: HookPoint, to_bwd_cache: Dict) -> Tensor:
            if hook.name == sae_hook_name(hook_position_end[0].name):
                # store activation for backward later
                to_bwd_cache[sae_hook_name(hook_position_end[0].name)] = act
                
            elif error_term_name(hook_position_end[0].name) == hook.name and self.use_error_term:
                # store activation for backward later
                to_bwd_cache[error_term_name(hook_position_end[0].name)] = act
                    
            return act
        
        try:
            with self._setup_error_term(self.use_error_term):
                for sae_name, sae in self.dict_saes.items():
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
            for sae in self.dict_saes.values():
                sae.reset_hooks()
    
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
            if hook.name == sae_hook_name(sae_name):
                if patch_deleted_comp and corrupt_cache is not None:
                    act_mask = self.nodes[(sae_name, (None,))].act == 0 # type: ignore
                    act[:, act_mask] = corrupt_cache[sae_hook_name(sae_name)][:, act_mask]
                fwd_cache[sae_hook_name(sae_name)] = act.detach()
                
            elif hook.name == error_term_name(sae_name) and self.use_error_term:
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
    def _setup_forward_model_hook(self, use_error_term: bool | None = None):
        use_error_term = use_error_term if use_error_term is not None else self.use_error_term
        try: 
            for sae in self._saes_to_list():
                self.model.add_sae(sae, use_error_term)
                
            yield
        finally:
            self.model.reset_saes()
                
    @contextmanager  
    def _setup_error_term(self, use_error_term: bool | None = None):
        if use_error_term is None:
            use_error_term = self.use_error_term
        
        orig_use_error_term = {}
        try: 
            for sae_name, sae in self.dict_saes.items():
                sae.use_error_term = use_error_term
                orig_use_error_term[sae_name] = sae.use_error_term
            yield
        finally:
            for sae_name, sae in self.dict_saes.items():
                sae.use_error_term = orig_use_error_term[sae_name]   
    
    @contextmanager
    def _detach_error_term(self, detach: bool, sae_name: str | None = None):
        orig_detach_error_term = {}
        orig_disable_error_grad = {}
        try:
            for name, sae in self.dict_saes.items():
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
            for name, sae in self.dict_saes.items():
                sae.detach_error_term = orig_detach_error_term[name]
                sae.disable_error_grad = orig_disable_error_grad[name]
                
    def save(self, filepath: str) -> None:
        """Saves the graph parameters, states, and metadata to a file."""
        # Extract hook names for validation later
        sae_hook_names = [sae.cfg.hook_name for sae in self.dict_saes.values()]
        
        state = {
            "metadata": {
                "model_name": self.model.cfg.model_name,
                "sae_hook_names": sae_hook_names,
            },
            "params": {
                "use_error_term": self.use_error_term,
                "seq_length": self.seq_length,
                "token_wise": self.token_wise,
            },
            "graph_state": {
                "connection": self.connection,
                "nodes": self.nodes,
                "node_scores": self.node_scores,
                "edges": self.edges,
                "edge_scores": self.edge_scores,
            }
        }
        t.save(state, filepath)

    @classmethod
    def load(
        cls, 
        filepath: str, 
        model: Any, 
        saes: Dict[int, List[Tuple[str, Any]]]
    ) -> "Feature_Graph":
        """Loads a graph from a file, validating model and SAE compatibility."""
        # Load the dictionary, mapping tensors to the current model's device
        state = t.load(filepath, map_location=model.cfg.device)
        
        # --- 1. Validation ---
        saved_model_name = state["metadata"]["model_name"]
        saved_sae_hooks = state["metadata"]["sae_hook_names"]
        
        if model.cfg.model_name != saved_model_name:
            raise ValueError(f"Model name mismatch! Saved: {saved_model_name}, Current: {model.cfg.model_name}")
            
        current_sae_hooks = []
        for layer in range(model.cfg.n_layers):
            if layer in saes:
                for _, sae in saes[layer]:
                    current_sae_hooks.append(sae.cfg.hook_name)
                
        # We use set() here so it checks that the hooks are similar regardless of minor reordering
        if set(current_sae_hooks) != set(saved_sae_hooks):
            raise ValueError(f"SAE hook mismatch!\nSaved: {saved_sae_hooks}\nCurrent: {current_sae_hooks}")
            
        # --- 2. Initialization ---
        # Initialize using the saved `use_error_term` to ensure the structure matches
        instance = cls(model=model, saes=saes, use_error_term=state["params"]["use_error_term"])
        
        # --- 3. Restore State ---
        instance.seq_length = state["params"]["seq_length"]
        instance.token_wise = state["params"]["token_wise"]
        
        graph_state = state["graph_state"]
        instance.connection = graph_state["connection"]
        instance.nodes = graph_state["nodes"]
        instance.node_scores = graph_state["node_scores"]
        instance.edges = graph_state["edges"]
        instance.edge_scores = graph_state["edge_scores"]
        
        return instance