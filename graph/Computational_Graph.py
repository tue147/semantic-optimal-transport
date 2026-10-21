import torch as t
from torch import Tensor
from .Graph_Template import Graph, GraphName, Node, Index
from typing import List, Tuple, Dict, Union, Callable
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
from .Graph_utils import nested_dict_to_string

def _tuple_to_act_name(tuple_name: Tuple) -> str:
    list_name = list(tuple_name)
    return utils.get_act_name(list_name[2], list_name[0], list_name[1])

class ComponentNode(Node):
    def __init__(
        self,
        name: Tuple[int|None, str|None, str], # (layer, layer_type, name)
    ):
        self._name = _tuple_to_act_name(name)
    
    @property
    def name(self) -> str:
        return self._name
    
    def __repr__(self) -> str:
        return self._name
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ComponentNode):
            return self._name == other.name
        elif isinstance(other, str):
            return self._name == other
        else:
            raise NotImplementedError("other is not an instance of ComponentNode or str")
    
    def __hash__(self) -> int:
        return hash(self._name)
    
class ComponentIndex(Index):
    def __init__(
        self,
        list_index: tuple[int|None, int|None, int|None] # [:, :, 0] --> [None, None, 0] (batch, seq, head)
    ):
        for index in list_index:
            assert type(index) == int or index == None, "index is not an instance of int or None"
        self.list_index = list_index
            
    @property
    def as_index(self) -> Tuple[int|slice, ...]:
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
        if isinstance(other, ComponentIndex):
            return self.list_index == other.list_index
        elif isinstance(other, tuple):
            return self.list_index == other
        else:
            raise NotImplementedError("other is not an instance of ComponentIndex or tuple")
    
    def __hash__(self) -> int:
        return hash(self.list_index)
    
    
class AttnIndex(ComponentIndex):
    def __init__(
        self,
        head_index: int
    ):
        super().__init__((None, None, head_index)) # (batch, seq, head, :)
        
class QkvIndex(ComponentIndex):
    def __init__(
        self,
        qkv_index: int
    ):
        super().__init__((None, None, qkv_index)) # (batch, seq, head, :)
        
class MlpIndex(ComponentIndex):
    def __init__(
        self,
        mlp_index: int
    ):
        super().__init__((None, None, None)) # (batch, seq, :)
        
class EmbedIndex(ComponentIndex):
    def __init__(
        self,
        index: int,
    ):
        super().__init__((None, None, None)) # (batch, seq, :)
        
class EndIndex(ComponentIndex):
    def __init__(
        self,
        index: int,
    ):
        super().__init__((None, None, None)) # (batch, seq, :)

# Helper function to create nested OrderedDicts
def nested_ordered_dict() -> defaultdict:
    return defaultdict(nested_ordered_dict)

# Convert defaultdict to OrderedDict (optional, for consistency)
def convert_edges_to_ordered_dict(dict: defaultdict) -> OrderedDict:
    def convert(d):
        if isinstance(d, defaultdict):
            return OrderedDict({k: convert(v) for k, v in d.items()})
        return d
    return convert(dict)

class Component_Graph(Graph):
    def __init__(
        self, 
        model: HookedTransformer,
    ):
        '''
        Computational graph of HookedTransformer
        Used for edge patching, node patching, and compute on the graph
        
        Data structure:
            
            self.edges: OrderedDict[
                ComponentNode, OrderedDict[
                    ComponentIndex, OrderedDict[
                        Node, OrderedDict[
                            ComponentIndex, float  # edge weight
                        ]
                    ]
                ]
            ]
            
            self.nodes: Dict[
                Tuple[Node, ComponentIndex], float  # node value
            ]
        '''
        
        self.model = model
        self.cfg = model.cfg
        assert not self.cfg.parallel_attn_mlp, "parallel attention and mlp mode is not supported"
        assert not self.cfg.attn_only, "attention only mode is not supported"
        assert self.cfg.use_attn_result, "use_attn_result should be True"
        
        self.n_layers = self.cfg.n_layers
        self.n_heads = self.cfg.n_heads
        self._reset_graph()
        
    def graph_type(self) -> str:
        return GraphName.computational_graph
        
    def get_graph(self) -> OrderedDict | defaultdict:
        self._check_graph()
        return self.edges
    
    def get_edge_value(
        self, 
        start_node: Node, 
        start_index: Index, 
        end_node: Node, 
        end_index: Index
    ) -> float | int | None:
        self._check_graph()
        return self.edges[end_node][end_index][start_node][start_index] 
    
    def get_nodes(self) -> Dict:
        self._check_graph()
        return self.nodes
    
    def get_node_value(self, node: Node, index: Index) -> float | int:
        self._check_graph()
        return self.nodes[(node, index)]
        
    def build_graph_from_graph(
        self,
        graph,
    ) -> OrderedDict | defaultdict:
        assert isinstance(graph, Component_Graph), "graph is not an instance of Graph"
        self._reset_graph()
        self.edges = deepcopy(graph.get_graph())
        self.nodes = deepcopy(graph.get_nodes())
        self._check_graph()
        return self.edges
    
    def add_node(
        self,
        node: Node,
        index: Index,
    ) -> None:
        self._check_graph()
        self._find_node(node, index, "add")
        self.nodes[(node, index)] = -1
        
    def update_node(
        self,
        node: Node,
        index: Index,
        value: float | int | Tensor
    ) -> None:
        self._check_graph()
        assert isinstance(value, float) or isinstance(value, int) or isinstance(value, Tensor), "value is not an instance of float or int or Tensor"
        self._find_node(node, index, "update")
        self.nodes[(node, index)] = value.item() if isinstance(value, Tensor) else value
    
    def delete_node(
        self,
        node: Node,
        index: Index,
    ) -> None:
        self._check_graph()
        self._find_node(node, index, "delete")
        # No need to delete the node from the nodes dict, or set to None
        
    def iterate_nodes(self) -> List[Tuple[Node, Index]]:
        self._check_graph()
        return list(self.nodes.keys())
    
    def find_deleted_nodes(self) -> List[Tuple[Node, Index]]:
        self._check_graph()
        all_nodes = set(self.nodes.keys())
        active_nodes = set()
        
        for end_node in self.edges:
            for end_index in self.edges[end_node]:
                for start_node in self.edges[end_node][end_index]:
                    for start_index in self.edges[end_node][end_index][start_node]:
                        if self.edges[end_node][end_index][start_node][start_index] is not None:
                            active_nodes.add((start_node, start_index))
                            
        return list(all_nodes - active_nodes)
        
    def _find_node(self, 
        node: Node, 
        index: Index,
        mode: str
    ) -> None:
        assert mode in ["add", "update", "delete"], "mode is not in ['add', 'update', 'delete']"
        found = False
        for end_node in self.edges:
            for end_index in self.edges[end_node]:
                for start_node in self.edges[end_node][end_index]:
                    for start_index in self.edges[end_node][end_index][start_node]:
                        if start_node == node and start_index == index: # only prune start node, avoid pruning qkv and end_node
                            found = True
                            if mode == "add":
                                self.edges[end_node][end_index][start_node][start_index] = -1
                            elif mode == "update":
                                return # skip the assertion, since the node is found
                            elif mode == "delete":
                                self.edges[end_node][end_index][start_node][start_index] = None
        assert found, "node is not found"
        
    # TODO: check if the edge already exists
    def add_edge(
        self,
        start_node: Node,
        start_index: Index,
        end_node: Node,
        end_index: Index
    ) -> None:
        self._check_graph()
        self.edges[end_node][end_index][start_node][start_index] = -1
        
    def update_edge(
        self,
        start_node: Node,
        start_index: Index,
        end_node: Node,
        end_index: Index,
        value: float | int | Tensor
    ) -> None:
        self._check_graph()
        assert isinstance(value, float) or isinstance(value, int) or isinstance(value, Tensor), "value is not an instance of float or int or Tensor"
        self.edges[end_node][end_index][start_node][start_index] = value.item() if isinstance(value, Tensor) else value
        
    def delete_edge(
        self,
        start_node: Node,
        start_index: Index,
        end_node: Node,
        end_index: Index
    ) -> None:
        self._check_graph()
        self.edges[end_node][end_index][start_node][start_index] = None
        
    def iterate_edges(self) -> List[Tuple[Node, Index, Node, Index]]:
        self._check_graph()
        edges = []
        for end_node in self.edges:
            for end_index in self.edges[end_node]:
                for start_node in self.edges[end_node][end_index]:
                    for start_index in self.edges[end_node][end_index][start_node]:
                        edges.append((start_node, start_index, end_node, end_index))
        return edges
                
    def find_deleted_edges(self) -> List[Tuple[Node, Index, Node, Index]]:
        self._check_graph()
        edges = []
        for end_node in self.edges:
            for end_index in self.edges[end_node]:
                for start_node in self.edges[end_node][end_index]:
                    for start_index in self.edges[end_node][end_index][start_node]:
                        if self.edges[end_node][end_index][start_node][start_index] is None:
                            edges.append((start_node, start_index, end_node, end_index))
        return edges
    
    def build_default_graph(
        self,
        attn: bool = True,
        qkv: bool = True,
        mlp: bool = True,
        embed: bool = True,
    ) -> OrderedDict | defaultdict:
        assert attn or qkv or mlp, "attn, kqv, mlp are all False"
        if qkv:
            assert attn, "qkv is True but attn is False"
        self._reset_graph()
        self.attn = attn
        self.qkv = qkv
        self.mlp = mlp
        self.embed = embed
        self.end_node = ComponentNode((self.n_layers-1, None, "resid_post"))
        self._build_graph()
        return self.edges
        
    def _build_graph(self) -> None:
        self._check_build_default_graph()
        for layer in range(0, self.n_layers):
            if self.attn:
                for head in range(self.n_heads):
                    if self.qkv:
                        self._add_default_edge(layer, None, "q_input", AttnIndex(head))
                        self._add_default_edge(layer, None, "k_input", AttnIndex(head))
                        self._add_default_edge(layer, None, "v_input", AttnIndex(head))
                    else:
                        self._add_default_edge(layer, None, "attn_in", AttnIndex(head))
            if self.mlp:
                self._add_default_edge(layer, None, "mlp_in", MlpIndex(-1))
        self._add_default(self.n_layers, self.end_node, EndIndex(-1)) # type: ignore
        self.edges = convert_edges_to_ordered_dict(self.edges) # type: ignore
        self._build_nodes_from_edges()
                
    def _add_default_edge(
        self,
        layer: int,
        layer_type: str|None,
        name: str,
        index: Index,
    ) -> None:
        end_node = ComponentNode((layer, layer_type, name))
        self._add_default(layer, end_node, index)
    
    def _add_default(
        self,
        layer: int,
        end_node: Node,
        index: Index,
    ) -> None:
        # Token embedding and positional embedding
        if self.embed:
            self.edges[end_node][index][ComponentNode((None, None, "embed"))][EmbedIndex(-1)] = -1
            self.edges[end_node][index][ComponentNode((None, None, "pos_embed"))][EmbedIndex(-1)] = -1
        # Attn and MLP from previous layers
        for prev_layer in range(0, layer):  # from 0 to layer-1
            if self.attn:
                for head in range(self.n_heads):
                    self.edges[end_node][index][ComponentNode((prev_layer, None, "result"))][AttnIndex(head)] = -1
            if self.mlp:
                self.edges[end_node][index][ComponentNode((prev_layer, None, "mlp_out"))][MlpIndex(-1)] = -1
        # Attn -> Mlp at the same layer
        if isinstance(index, MlpIndex):
            for head in range(self.n_heads):
                self.edges[end_node][index][ComponentNode((layer, None, "result"))][AttnIndex(head)] = -1
                
    def _build_nodes_from_edges(self) -> None: # only build for start node, avoid building for qkv and end_node
        for end_node in self.edges:
            for end_index in self.edges[end_node]:
                for start_node in self.edges[end_node][end_index]:
                    for start_index in self.edges[end_node][end_index][start_node]:
                        self.nodes[(start_node, start_index)] = -1
    
    def _reset_graph(self) -> None:
        # Initialize edges as a nested defaultdict for automatic creation of OrderedDict levels
        self.edges = nested_ordered_dict()
        self.nodes = OrderedDict()
        
        self.attn = None
        self.qkv = None
        self.mlp = None
        self.embed = None
        self.end_node = None
                
    def _check_build_default_graph(self) -> None:
        # Check if the attributes are specified to build the default graph
        assert isinstance(self.attn, bool), "the attn attribute is not specified"
        assert isinstance(self.qkv, bool), "the qkv attribute is not specified"
        assert isinstance(self.mlp, bool), "the mlp attribute is not specified"
        assert isinstance(self.embed, bool), "the embed attribute is not specified"
        assert isinstance(self.end_node, Node), "the end_node attribute is not specified"
        
    def _check_graph(self) -> None:
        # Check if the graph is built
        assert isinstance(self.edges, OrderedDict), "the graph is not built"
        assert len(self.nodes) > 0, "the nodes are not built"
    
    def model_setup(self) -> None:
        # Set up the model for the forward pass
        self.model.set_use_attn_in(True)
        self.model.set_use_attn_result(True)
        self.model.set_use_hook_mlp_in(True)
        self.model.set_use_split_qkv_input(True)
            
    def __repr__(self) -> str:
        self._check_graph()
        return nested_dict_to_string(self.edges, indent=4)
    
    def run_model(self, toks: Tensor) -> Tuple[Tensor, ActivationCache]:
        '''
        Run the model and return the logits and cache
        '''
        return self.model.run_with_cache(toks) # type: ignore
    
    def forward(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''
        Forward pass of the graph with clean tokens, if the edge exists, replace the activation with corrupted activation
        '''
        self.model.reset_hooks()
        self.model_setup()
        local_cache = {} # cache for the online activations
        
        def hook_fn(orig_tensor: Tensor, hook: HookPoint) -> Tensor:
            if hook.name in self.edges:
                for end_index in self.edges[hook.name]:
                    for start_node in self.edges[hook.name][end_index]:
                        for start_index in self.edges[hook.name][end_index][start_node]:
                            if self.edges[hook.name][end_index][start_node][start_index] is None and corrupt_cache is not None:
                                # in place operation for memory efficiency, cannot do this for backward
                                orig_tensor[end_index.as_index] += (
                                    corrupt_cache[start_node.name][start_index.as_index] - 
                                    local_cache[start_node.name][start_index.as_index]
                                )
            
            local_cache[hook.name] = orig_tensor # update the local cache
            return orig_tensor
        
        self.model.add_hook(lambda name: True, hook_fn) # type: ignore
        
        with t.no_grad():
            logits = self.model(clean_token)
        self.model.reset_hooks()
    
        return logits, local_cache
    
    def forward_backward_gradient(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        metric: Callable[[Tensor], Tensor],
        show_warnings: bool = True,
        retain_graph: bool = False,
        mode: str | None = None,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], Tensor],  # node effects
        Dict[Tuple[Node, Index, Node, Index], Tensor], # edge effects
    ]:
        assert mode == "node" or mode == "edge" or mode == None, "mode is not in ['node', 'edge', None]"
        node_grads, edge_grads, clean_cache = self._forward_backward_gradient(
            clean_token, corrupt_cache, metric, show_warnings, retain_graph, **kwargs
        )
        return_node = True if mode == "node" or mode is None else False
        return_edge = True if mode == "edge" or mode is None else False
        node_effect = self._attib_effect(node_grads, corrupt_cache, clean_cache, self.iterate_nodes) if return_node else {}
        edge_effect = self._attib_effect(edge_grads, corrupt_cache, clean_cache, self.iterate_edges) if return_edge else {}
        return node_effect, edge_effect
        
    def _attib_effect(
        self,
        grads: Dict,
        corrupt_cache: ActivationCache | Dict | None,
        clean_cache: ActivationCache | Dict,
        iterative_handler: Callable[[], List[Tuple[Node, Index]] | List[Tuple[Node, Index, Node, Index]]],
    ) -> Dict:
        attrib_effect = {} 
        for comp in iterative_handler():
            if corrupt_cache is not None:
                attrib_effect[comp] = (
                    grads[comp] * 
                    (corrupt_cache[comp[0].name][comp[1].as_index] - clean_cache[comp[0].name][comp[1].as_index])
                ).sum()
            else:
                attrib_effect[comp] = (
                    grads[comp] * 
                    (0 - clean_cache[comp[0].name][comp[1].as_index])
                ).sum()
        
        return attrib_effect
    
    def _forward_backward_gradient(
        self,
        clean_token: Tensor,
        corrupt_cache: ActivationCache | Dict[str, Tensor] | None,
        metric: Callable[[Tensor], Tensor],
        show_warnings: bool = True,
        retain_graph: bool = False,
        **kwargs,
    ) -> Tuple[
        Dict[Tuple[Node, Index], Tensor],  # node gradients
        Dict[Tuple[Node, Index, Node, Index], Tensor], # edge gradients
        Dict[str, Tensor] # activation cache
    ]:
        '''
        Forward pass of the graph with clean tokens, if the edge exists, replace the activation with corrupted activation
        Backward pass on the graph wrt the metric
        Return the gradients wrt nodes, edges, and activation cache
        '''
        self.model.reset_hooks()
        self.model_setup()
        first_warning_shown = False
        
        local_cache = {} # cache for the online activations
        def hook_fn(orig_tensor: Tensor, hook: HookPoint) -> Tensor:
            nonlocal first_warning_shown
            # not using in place operation for backward
            modified_tensor = orig_tensor.clone()
            
            if hook.name in self.edges:
                for end_index in self.edges[hook.name]:
                    for start_node in self.edges[hook.name][end_index]:
                        for start_index in self.edges[hook.name][end_index][start_node]:
                            if self.edges[hook.name][end_index][start_node][start_index] is None:
                                if corrupt_cache is not None:
                                    modified_tensor[end_index.as_index] = (
                                        modified_tensor[end_index.as_index] +
                                        corrupt_cache[start_node.name][start_index.as_index].detach() -
                                        local_cache[start_node.name][start_index.as_index]
                                    )
                                else:
                                    modified_tensor[end_index.as_index] = (
                                        modified_tensor[end_index.as_index] -
                                        local_cache[start_node.name][start_index.as_index]
                                    )
                                # show the warning only once
                                if not first_warning_shown and show_warnings:
                                    warn(
                                        '''
                                        Warning: If edges are deleted, the gradient approximation may be inaccurate.
                                        This is due to inplace modification of "corrupted" activations during forward pass.
                                        ''',
                                        UserWarning,
                                    )
                                    first_warning_shown = True
            
            local_cache[hook.name] = modified_tensor # update the local cache
            return modified_tensor
        
        bwd_cache = {}
        def hook_fn_bwd(grad: Tensor, hook: HookPoint):
            bwd_cache[hook.name] = grad.detach()
            
        with t.set_grad_enabled(True):
            with self.model.hooks(
                fwd_hooks=[(lambda name: True, hook_fn)],
                bwd_hooks=[(lambda name: True, hook_fn_bwd)]
            ):
                logits = self.model(clean_token)
                loss = metric(logits)
                loss.backward(retain_graph=retain_graph)
        
        node_grads = {}
        for node in self.nodes:
            node_grads[node] = bwd_cache[node[0].name][node[1].as_index]
            
        edge_grads = {}
        for edge in self.edges:
            for end_index in self.edges[edge]:
                for start_node in self.edges[edge][end_index]:
                    for start_index in self.edges[edge][end_index][start_node]:
                        # gradient of the edge is the gradient of the end node wrt the start node 
                        # due to the "add" operation in the forward pass
                        edge_grads[(start_node, start_index, edge, end_index)] = bwd_cache[edge.name][end_index.as_index]
                        
        self.model.reset_hooks()           
        return node_grads, edge_grads, local_cache
        
               
    def __call__(
        self, 
        clean_token: Tensor,
        corrupt_cache: ActivationCache,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self.forward(clean_token, corrupt_cache)



if __name__ == "__main__":
    # '''
    # For computational graph testing
    # '''
    # device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    # gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)
    # gpt2_small.set_use_attn_result(True)
    # graph = Component_Graph(gpt2_small)
    # graph.build_default_graph(attn=True, qkv=True, mlp=True, embed=True)
    # # print(graph)
    # print(graph.iterate_nodes())
    # # print(graph.iterate_edges())
    # graph2 = Component_Graph(gpt2_small)
    # graph2.build_graph_from_graph(graph)
    # nodes_to_delete = [
    #     (ComponentNode((11, None, "mlp_out")), MlpIndex(-1)),
    #     (ComponentNode((5, None, "result")), AttnIndex(5)),
    #     (ComponentNode((8, None, "result")), AttnIndex(6)),
    #     (ComponentNode((1, None, "mlp_out")), MlpIndex(-1)),
    #     (ComponentNode((0, None, "mlp_out")), MlpIndex(-1)),
    #     (ComponentNode((9, None, "result")), AttnIndex(10)),
    #     # (ComponentNode((None, None, "embed")), EmbedIndex(-1)),
    # ]
    # for node in nodes_to_delete:
    #     graph2.delete_node(*node)
    # graph2.delete_edge(
    #     ComponentNode((2, None, "mlp_out")), MlpIndex(-1), ComponentNode((3, None, "q_input")), AttnIndex(0)
    # )
    # # graph2.update_node(ComponentNode((0, None, "mlp_out")), MlpIndex(-1), 1)
    # # print(graph2)
    # # print(graph2.get_nodes()[(ComponentNode((0, None, "mlp_out")), MlpIndex(-1))])
    
    # clean_data = "hello, my name is T"
    # corrupt_data = "hi, his name is T"
    # clean_token = gpt2_small.to_tokens(clean_data)
    # corrupt_token = gpt2_small.to_tokens(corrupt_data)
    
    # corrupt_logit, corrupt_cache = gpt2_small.run_with_cache(corrupt_token)
    # logits, patched_cache = graph2.forward(clean_token, corrupt_cache)
    
    # gpt2_small.reset_hooks()
    
    # def hook_fn(orig_tensor: Tensor, hook: HookPoint) -> Tensor:
    #     for node in nodes_to_delete:
    #         if hook.name == node[0]:
    #             orig_tensor[node[1].as_index] = corrupt_cache[node[0].name][node[1].as_index]
    #             break
    #     return orig_tensor
    
    # gpt2_small.add_hook(lambda name: True, hook_fn) # type: ignore
    # logits2, _ = gpt2_small.run_with_cache(clean_token)
    # t.testing.assert_close(logits, logits2, rtol=0.01, atol=1e-04)  
    
    
    '''
    For gradient testing
    '''
    device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)
    gpt2_small.set_use_attn_result(True)
    graph = Component_Graph(gpt2_small)
    graph.build_default_graph(attn=True, qkv=True, mlp=True, embed=True)
    nodes_to_delete = [
        # (ComponentNode((11, None, "mlp_out")), MlpIndex(-1)),
        # (ComponentNode((5, None, "result")), AttnIndex(5)),
        # (ComponentNode((10, None, "result")), AttnIndex(10)),
        # (ComponentNode((9, None, "result")), AttnIndex(5)),
        # (ComponentNode((9, None, "result")), AttnIndex(9)),
        # (ComponentNode((8, None, "result")), AttnIndex(6)),
        # (ComponentNode((1, None, "mlp_out")), MlpIndex(-1)), 
        # (ComponentNode((0, None, "mlp_out")), MlpIndex(-1)),
        # (ComponentNode((0, None, "result")), AttnIndex(10)),
    ]
    for node in nodes_to_delete:
        graph.delete_node(*node)

    N = 25
    from ioi_dataset import IOIDataset
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=gpt2_small.tokenizer,
        prepend_bos=False,
        seed=1,
        device=str(device),
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    
    def logits_to_ave_logit_diff(
        logits: Tensor,  # "batch seq d_vocab" 
        ioi_dataset: IOIDataset, 
        per_prompt: bool = False,
        reduction: str = "mean",
    ) -> Tensor: # "batch"
        """
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        """
        # Only the final logits are relevant for the answer
        # Get the logits corresponding to the indirect object / subject tokens respectively
        io_logits: Tensor = logits[  # "batch"
            range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
        ]
        s_logits: Tensor = logits[   # "batch"
            range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
        ]
        # Find logit difference
        answer_logit_diff = io_logits - s_logits
        if reduction == "mean":
            reduction_fn = t.mean
        elif reduction == "sum":
            reduction_fn = t.sum
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        return answer_logit_diff if per_prompt else reduction_fn(answer_logit_diff)
    
    ioi_logits_original, ioi_cache = gpt2_small.run_with_cache(ioi_dataset.toks)
    abc_logits_original, abc_cache = gpt2_small.run_with_cache(abc_dataset.toks)
    
    ioi_average_logit_diff = logits_to_ave_logit_diff(ioi_logits_original, ioi_dataset, reduction="mean") # type: ignore
    abc_average_logit_diff = logits_to_ave_logit_diff(abc_logits_original, ioi_dataset, reduction="mean") # type: ignore
    
    def ioi_metric(
        logits: Tensor, # "batch seq d_vocab"
        clean_logit_diff: Tensor = ioi_average_logit_diff,
        corrupted_logit_diff: Tensor = abc_average_logit_diff,
        ioi_dataset: IOIDataset = ioi_dataset,
    ) -> Tensor: # scalar float
        """
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        """
        logit_diff = logits_to_ave_logit_diff(logits, ioi_dataset, reduction="mean")
        return (logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

    clean_token = ioi_dataset.toks
    corrupt_cache = abc_cache
    metrics = ioi_metric
    
    gpt2_small.reset_hooks()
    node_effects, edge_effects = graph.forward_backward_gradient(clean_token, corrupt_cache, metrics)
    gpt2_small.reset_hooks()

    # NOTE: uncomment this block to calculate the attribution effect on NODES
    attrib_effect_unprocessed = node_effects 
    ablation_effect = {}
    clean_metric = metrics(graph(clean_token, corrupt_cache)[0])
    for node, index in graph.iterate_nodes():
        graph.delete_node(node, index)
        patched_logits, _ = graph.forward(clean_token, corrupt_cache)
        ablation_effect[node.name + repr(index)] = metrics(patched_logits).item() - clean_metric.item()
        if (node, index) not in nodes_to_delete:
            graph.add_node(node, index)
    attrib_effect = {}
    for node, index in attrib_effect_unprocessed:
        attrib_effect[node.name + repr(index)] = attrib_effect_unprocessed[(node, index)].cpu().item() # type: ignore

    # # NOTE: uncomment this block to calculate the attribution effect on EDGES
    # list_keys = list(edge_effects.keys())[50:100]
    # attrib_effect_unprocessed = {key: edge_effects[key] for key in list_keys}
    # ablation_effect = {}
    # clean_metric = metrics(graph(clean_token, corrupt_cache)[0])
    # for edge in list_keys:
    #     graph.delete_edge(*edge)
    #     patched_logits, _ = graph.forward(clean_token, corrupt_cache)
    #     ablation_effect[edge[0].name + repr(edge[1]) + edge[2].name + repr(edge[3])] = metrics(patched_logits).item() - clean_metric.item()
    #     if (edge[0], edge[1]) not in nodes_to_delete:
    #         graph.add_edge(*edge)
    # attrib_effect = {}
    # for edge in attrib_effect_unprocessed:
    #     attrib_effect[
    #         edge[0].name + repr(edge[1]) + edge[2].name + repr(edge[3])
    #     ] = attrib_effect_unprocessed[edge].cpu().item() # type: ignore
        
    from plotly import express as px
    import pandas as pd
    df = pd.DataFrame({
        'Keys': list(attrib_effect.keys()), 
        'ablation': list(ablation_effect.values()), 
        'attribution': list(attrib_effect.values()),
    })

    # Create scatter plot
    fig = px.scatter(df, x='attribution', y='ablation', text='Keys', labels={'attribution': 'attribution', 'ablation': 'ablation'},
                    title='Comparison between attribution and ablation patching')
    fig.update_traces(textposition='top center')

    # Add y = x line
    fig.add_shape(type='line', x0=min(df['ablation']), y0=min(df['ablation']), 
                x1=max(df['ablation']), y1=max(df['ablation']),
                line=dict(color='red', dash='dash'))
    fig.show()
    
    
    # Compute ablation - attribution
    df['ablation_minus_attribution'] = df['ablation'] - df['attribution']

    # Create bar plot
    fig = px.bar(df, x='Keys', y='ablation_minus_attribution', text='ablation_minus_attribution',
                labels={'ablation_minus_attribution': 'Ablation - Attribution'},
                title='Difference between Ablation and Attribution')

    fig.update_traces(textposition='outside')
    fig.show()