import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from functools import partial
from typing import List, Tuple, Dict, Any, Union, Callable, Literal
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
)
import circuitsvis as cv
from itertools import product
import random
from copy import deepcopy
from collections import OrderedDict, defaultdict
from transformer_lens import ActivationCache
from graph.Graph_Template import Graph, GraphName
from graph.Sparse_act import SparseAct

class Pruner:
    def __init__(
        self,
        graph: Graph,
        metric: Callable[[Tensor], Tensor],
        device: t.device,
        verbose: bool = False,
    ):
        '''
        Pruner class for pruning the model.
        
        Args:
            graph: Graph, the computational graph of the model.
            model: nn.Module, the model to be pruned.
            metric: Callable[[Tensor], Tensor], the metric to be used for pruning. Takes logits as input and returns a scalar.
            device: t.device, the device to be used for pruning.
            verbose: bool, whether to print the pruning process.
        '''
        self.graph = graph
        self.metric = metric
        self.device = device
        self.verbose = verbose  
        self.effects: Dict[Any, Any] = {}
    
    def __call__(self,
        clean_tokens: Tensor,
        corrupt_tokens_or_cache: Tensor | Dict[str, Tensor] | ActivationCache | None,
        threshold: float,
        prune_type: str = "attrib",
        cut_mode: str = "node",
        scoring_mode: str = "abs",
        threshold_type: str = "value",
        modify_inplace: bool = False,
        return_type: str | None = "retained",
        reverse_pruning: bool = False,
        **kwargs,
    ) -> Union[Graph, Tuple[Graph, Dict]]:
        return self.prune(clean_tokens, corrupt_tokens_or_cache, threshold, prune_type, cut_mode, scoring_mode, threshold_type, modify_inplace, return_type, reverse_pruning, **kwargs)
    
    def prune(
        self,
        clean_tokens: Tensor,
        corrupt_tokens_or_cache: Tensor | Dict[str, Tensor] | ActivationCache | None,
        threshold: float,
        prune_type: str = "attrib",
        cut_mode: str = "node",
        scoring_mode: str = "abs",
        threshold_type: str = "value",
        modify_inplace: bool = False,
        return_type: str | None = "retained",
        reverse_pruning: bool = False,
        **kwargs,
    ) -> Union[Graph, Tuple[Graph, Dict]]:
        '''
        Prune the graph.
        Args:
            clean_tokens: Tensor, the clean tokens for the model.
            corrupt_tokens_or_cache: Tensor | Dict[str, Tensor] | ActivationCache, the corrupt tokens or corrupt cache for the model.
            threshold: float, the threshold for pruning.
            prune_type: str, the type for pruning; only "attrib" (attribution patching) is implemented.
            cut_mode: str, the mode for pruning, either "node" or "edge".
            scoring_mode: str, the mode for scoring, either "abs" or "greater" or "less". 
                E.g. if scoring_mode is "greater" and threshold_type is "value, then the nodes/edges with metric greater than the baseline by a threshold will be pruned.
            threshold_type: str, the type of thresholding, either "value" or "percen" or "number".
                If "value": the nodes/edges with score lower (or higher - based on scoring_mode) will be pruned.
                If "percen": a certain proportion of nodes/edges will be retained.
                If "number": retain a certain number of nodes/edges.
            modify_inplace: bool, whether to modify the graph in place or return a new graph.
            return_type: str | None, the type to return, either "retained" or "all" or "None" or None.
            reverse_pruning: bool, whether to reverse the pruning.
            **kwargs: additional kwargs to pass to the forward_backward_gradient method.
        '''
        assert cut_mode in ["node", "edge"], f"mode must be either 'node' or 'edge', got {cut_mode}"
        assert scoring_mode in ["abs", "greater", "less"], f"scoring_mode must be either 'abs' or 'greater' or 'less', got {scoring_mode}"
        assert threshold_type in ["value", "percen", "number"], f"threshold_type must be either 'value' or 'percen' or 'number', got {threshold_type}"
        if threshold_type == "percen":
            assert threshold >= 0 and threshold <= 1
        if isinstance(corrupt_tokens_or_cache, Tensor):
            assert len(clean_tokens) == len(corrupt_tokens_or_cache) or len(corrupt_tokens_or_cache) == 1, "clean_tokens and corrupt_tokens must have the same length, or corrupt_tokens must have length 1."
        assert return_type in ["retained", "all", "None", None], f"return_type must be either 'retained' or 'all' or 'None', got {return_type}"
        
        if prune_type == "attrib":
            return self._prune_attributing(clean_tokens, corrupt_tokens_or_cache, threshold, cut_mode, "attrib", scoring_mode, threshold_type, modify_inplace, return_type, reverse_pruning, **kwargs)
        else:
            raise ValueError(f"Prune type {prune_type} is not implemented.")
         
    def _prune_attributing(
        self,
        clean_tokens: Tensor,
        corrupt_tokens_or_cache: Tensor | Dict[str, Tensor] | ActivationCache | None,
        threshold: float,
        cut_mode: str = "node",
        gradient_method: str = "attrib",
        scoring_mode: str = "abs",
        threshold_type: str = "value",
        modify_inplace: bool = False,
        return_type: str | None = "retained",
        reverse_pruning: bool = False,
        **kwargs, # additional kwargs to pass to the forward_backward_gradient method
    ) -> Union[Graph, Tuple[Graph, Dict]]:
        
        assert gradient_method in ["attrib"], f"gradient_method must be 'attrib', got {gradient_method}"
        if modify_inplace:
            graph = self.graph
        else:
            graph = deepcopy(self.graph)
            
        add_handler, update_handler, delete_handler, iterate_handler, find_deleted_handler = self._get_handler(graph, cut_mode)
        clean_tokens = clean_tokens.to(self.device)
        
        graph.model_setup() # set up the model before caching
        with t.no_grad(): # no gradient needed, save memory
            if isinstance(corrupt_tokens_or_cache, Tensor):
                _, corrupt_cache = graph.run_model(corrupt_tokens_or_cache.to(self.device))
            else:
                corrupt_cache = corrupt_tokens_or_cache
            clean_logits, _ = graph(clean_tokens, corrupt_cache)  # if the graph is pruned before, this will be patched_logits
            current_score = self.metric(clean_logits)
            if self.verbose:
                print(f"Current score: {current_score}")
            
        retained_components = {}
        all_components = {}
        if self.effects.get(cut_mode, None) is None:
            with t.set_grad_enabled(True):
                if gradient_method == "attrib":
                    node_effect, edge_effect = graph.forward_backward_gradient(
                        clean_tokens,
                        corrupt_cache,
                        self.metric,
                        show_warnings=True,
                        mode=cut_mode,
                        **kwargs,
                    )
                    
                    attrib_effect = node_effect if cut_mode == "node" else edge_effect
                    self.effects[cut_mode] = attrib_effect # save the effects so that we don't need to recompute them
                else:
                    raise ValueError(f"Gradient method {gradient_method} is not implemented.")
        else:
            attrib_effect = self.effects[cut_mode]
            
        if threshold_type == "value":
            for component in iterate_handler():
                # we don't need to pass the current score here, because:
                # attrib_effect[component] approximate: metric(corrupted) - metric(clean)
                retain, value = self._value_threshold(
                    0,
                    attrib_effect[component], 
                    threshold, 
                    scoring_mode,
                    reverse_pruning,
                )
                
                all_components[component] = value
                if retain:
                    update_handler(*(component + (value,)))
                    retained_components[component] = value
                else:
                    delete_handler(*component)
                    if cut_mode == "node":
                        update_handler(*(component + (value,)))
                    if self.verbose:
                        print(f"Pruned {component} with value {value}")
                        
        else:
            list_effects = []
            for component in iterate_handler():
                list_effects.append(attrib_effect[component])
                
            list_retain, list_value = self._number_and_percen_threshold(
                list_effects,
                threshold,
                scoring_mode,
                threshold_type,
                reverse_pruning,
            )
            
            for component, retain, value in zip(iterate_handler(), list_retain, list_value):
                all_components[component] = value
                if retain:
                    update_handler(*(component + (value,)))
                    retained_components[component] = value
                else:
                    delete_handler(*component)
                    if cut_mode == "node":
                        update_handler(*(component + (value,)))
                    if self.verbose:
                        print(f"Pruned {component} with value {value}")
                        
        if return_type == "retained":
            return graph, retained_components
        elif return_type == "all":
            return graph, all_components
        return graph
    
    def _value_threshold_sparse_coo(
        self,
        current_score: Tensor | float,
        temp_score: Tensor,
        threshold: float,
        scoring_mode: str,
        reverse_pruning: bool,
    ) -> Tuple[Any, Any]:
        if isinstance(current_score, float):
            current_score = t.sparse_coo_tensor(
                temp_score.indices(), t.full_like(temp_score.values(), current_score), temp_score.size()
            ).coalesce().values()

        if scoring_mode == "abs":
            value = (current_score - temp_score.values()).abs()
            retain_mask = value > threshold
        elif scoring_mode == "greater":
            value = temp_score.values() - current_score
            retain_mask = value < threshold
        else:
            value = temp_score.values() - current_score
            retain_mask = value > threshold
            
        if reverse_pruning:
            retain_mask = ~retain_mask

        value_sparse = t.sparse_coo_tensor(temp_score.indices(), value, temp_score.size()).coalesce()
        retained_sparse = t.sparse_coo_tensor(temp_score.indices(), retain_mask, temp_score.size()).coalesce()
        return True, (value_sparse, retained_sparse)

    def _value_threshold_dense(
        self,
        current_score: Tensor | SparseAct | float,
        temp_score: Tensor | SparseAct,
        threshold: float,
        scoring_mode: str,
        reverse_pruning: bool,
    ) -> Tuple[Any, Any]:
        if scoring_mode == "abs":
            value = (current_score - temp_score).abs()
            retain = value >= threshold
        elif scoring_mode == "greater":
            value = temp_score - current_score
            retain = value <= threshold
        else:
            value = temp_score - current_score
            retain = value >= threshold
            
        if reverse_pruning:
            retain = ~retain

        if retain.numel() == 1:
            return retain.item(), value
        else:
            return True, (value, retain)

    def _value_threshold(
        self,
        current_score: Tensor | SparseAct | float,
        temp_score: Tensor | SparseAct,
        threshold: float,
        scoring_mode: str,
        reverse_pruning: bool,
    ) -> Tuple[Any, Any]:
        with t.no_grad():
            if (
                isinstance(temp_score, Tensor) and temp_score.is_sparse
            ) or (
                isinstance(current_score, Tensor) and current_score.is_sparse
            ):
                return self._value_threshold_sparse_coo(current_score, temp_score, threshold, scoring_mode, reverse_pruning) # type: ignore
            else:
                return self._value_threshold_dense(current_score, temp_score, threshold, scoring_mode, reverse_pruning)
            
    def _number_and_percen_threshold_sparse_coo(
        self,
        list_scores: List[Tensor],
        threshold: float,
        scoring_mode: str,
        threshold_type: str,
        reverse_pruning: bool,
    ) -> Tuple[List[Any], List[Any]]:
        list_numel = []
        list_shapes = []
        for score in list_scores:
            list_numel.append(score.values().numel())
            list_shapes.append(score.values().shape)
            
        values = t.cat(
            [score.values().flatten() for score in list_scores], 
            dim=0,
        )
        
        num_retain_elements = int(threshold * sum(list_numel)) if threshold_type == "percen" else int(threshold)
            
        if scoring_mode == "abs":
            values = values.abs()
        # A NaN score makes CUDA topk trip `writeIndex < outputSliceSize` (device-side assert that
        # kills the job -- seen on 200-node circuits). Push NaNs to the never-retained end; prompts
        # without NaNs select exactly the same elements as before.
        if t.isnan(values).any():
            print(f"[Pruner] {int(t.isnan(values).sum())} NaN scores replaced before topk")
            values = t.nan_to_num(values, nan=float("inf") if scoring_mode == "greater" else float("-inf"))

        if scoring_mode == "abs":
            indices = t.topk(values.cpu(), k=num_retain_elements, dim=0, largest=True, sorted=False)[1]
        elif scoring_mode == "greater":
            indices = t.topk(values.cpu(), k=num_retain_elements, dim=0, largest=False, sorted=False)[1] # prune the top highest score = retain the lowest score
        else:
            indices = t.topk(values.cpu(), k=num_retain_elements, dim=0, largest=True, sorted=False)[1]

        # topk runs on CPU: torch 2.11's CUDA gatherTopK trips `writeIndex < outputSliceSize` on some
        # Gemma 200-node score vectors even without NaNs. Same retained set unless the cut is a tie.
        indices = indices.to(values.device)
        retains = t.zeros(values.size(0), dtype=t.bool, device=values.device)
        retains[indices] = True
            
        if reverse_pruning:
            retains = ~retains

        list_retains = []
        last_idx = 0
        for numel, shape, score in zip(list_numel, list_shapes, list_scores):
            retain = retains[last_idx:last_idx+numel].reshape(shape)
            list_retains.append(
                t.sparse_coo_tensor(score.indices(), retain, score.size()).coalesce()
            )
            last_idx += numel
            
        return [True for _ in list_retains], [(score, mask) for score, mask in zip(list_scores, list_retains)]
    
    def _number_and_percen_threshold_dense(
        self,
        list_scores: List[Tensor | SparseAct],
        threshold: float,
        scoring_mode: str,
        threshold_type: str,
        reverse_pruning: bool,
    ) -> Tuple[List[Any], List[Any]]:
        list_numel = []
        list_shapes = []
        for score in list_scores:
            list_numel.append(score.numel())
            list_shapes.append(score.shape if isinstance(score, Tensor) else score.to_tensor().shape)
            
        values = t.cat(
            [score.flatten() if isinstance(score, Tensor) else score.to_tensor().flatten() for score in list_scores], 
            dim=0,
        )
        
        num_retain_elements = int(threshold * sum(list_numel)) if threshold_type == "percen" else int(threshold)
            
        if scoring_mode == "abs":
            values = values.abs()
        # A NaN score makes CUDA topk trip `writeIndex < outputSliceSize` (device-side assert that
        # kills the job -- seen on 200-node circuits). Push NaNs to the never-retained end; prompts
        # without NaNs select exactly the same elements as before.
        if t.isnan(values).any():
            print(f"[Pruner] {int(t.isnan(values).sum())} NaN scores replaced before topk")
            values = t.nan_to_num(values, nan=float("inf") if scoring_mode == "greater" else float("-inf"))

        if scoring_mode == "abs":
            indices = t.topk(values.cpu(), k=num_retain_elements, dim=0, largest=True, sorted=False)[1]
        elif scoring_mode == "greater":
            indices = t.topk(values.cpu(), k=num_retain_elements, dim=0, largest=False, sorted=False)[1] # prune the top highest score = retain the lowest score
        else:
            indices = t.topk(values.cpu(), k=num_retain_elements, dim=0, largest=True, sorted=False)[1]

        # topk runs on CPU: torch 2.11's CUDA gatherTopK trips `writeIndex < outputSliceSize` on some
        # Gemma 200-node score vectors even without NaNs. Same retained set unless the cut is a tie.
        indices = indices.to(values.device)
        retains = t.zeros(values.size(0), dtype=t.bool, device=values.device)
        retains[indices] = True
            
        if reverse_pruning:
            retains = ~retains

        list_retains = []
        last_idx = 0
        for numel, shape, score in zip(list_numel, list_shapes, list_scores):
            retain = retains[last_idx:last_idx+numel].reshape(shape)
            list_retains.append(retain if isinstance(score, Tensor) else score.to_sparse_like_self(retain))
            last_idx += numel
            
        if list_retains[0].numel() == 1:
            return [retain.item() for retain in list_retains], list_scores
        else:
            return [True for _ in list_retains], [(score, mask) for score, mask in zip(list_scores, list_retains)]
            
    def _number_and_percen_threshold(
        self,
        list_scores: List[Tensor | SparseAct],
        threshold: float,
        scoring_mode: str,
        threshold_type: str,
        reverse_pruning: bool,
    ) -> Tuple[List[Any], List[Any]]:
        with t.no_grad():
            if isinstance(list_scores[0], Tensor) and list_scores[0].is_sparse:
                return self._number_and_percen_threshold_sparse_coo(list_scores, threshold, scoring_mode, threshold_type, reverse_pruning) # type: ignore
            else:
                return self._number_and_percen_threshold_dense(list_scores, threshold, scoring_mode, threshold_type, reverse_pruning)
        
    def _get_handler(
        self,
        graph: Graph,  # not always self.graph
        cut_mode: str,
    ) -> Tuple[Callable, Callable, Callable, Callable, Callable]:
        if cut_mode == "node":
            add_handler = graph.add_node
            update_handler = graph.update_node
            delete_handler = graph.delete_node
            iterate_handler = graph.iterate_nodes
            find_deleted_handler = graph.find_deleted_nodes
        else:
            add_handler = graph.add_edge
            update_handler = graph.update_edge
            delete_handler = graph.delete_edge
            iterate_handler = graph.iterate_edges
            find_deleted_handler = graph.find_deleted_edges
        return add_handler, update_handler, delete_handler, iterate_handler, find_deleted_handler
    