import os
import torch
import torch as t
import html
import json
import math
import einops
import re
from typing import Any, Dict, List, Tuple, OrderedDict, Optional, Literal
from torch import Tensor
from sae.SAE_Tools import *
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from llm_interpret_circuit_util import format_features_for_llm_clustering, call_llm_for_clustering, get_llm_clustering_prompt
import random

import ot 
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

def _render_token_html(tokens: List[str], acts: List[float], target_idx: int, max_global: float) -> str:
    html_str = ""
    for i, (tok, act) in enumerate(zip(tokens, acts)):
        ratio = act / max_global if max_global > 0 else 0
        alpha = max(0.0, min(1.0, ratio))
        bg_color = f"rgba(0, 200, 83, {alpha})" if act > 0 else "transparent"
        safe_tok = html.escape(tok).replace('\n', '↵')
        is_target = (i == target_idx)
        target_class = "target-token" if is_target else ""
        html_str += f'<span class="token {target_class}" style="background-color: {bg_color};">{safe_tok}<span class="tooltip">{act:.4f}</span></span>'
    return html_str

def generate_interactive_html_embeddable(data: Dict[str, Any]) -> str:
    # UPDATED CSS: margin-bottom and background added to example-row for white gaps
    css_content = """<style>
        .interval-block { border-bottom: 1px solid #eee; margin-bottom: 8px; padding-bottom: 3px; }
        .interval-header { background: #f1f3f5; padding: 2px 6px; font-size: 0.65rem; font-weight: bold; color: #555; display: flex; justify-content: space-between; border-radius: 2px; }
        .example-row { display: flex; align-items: flex-start; padding: 4px 0; border-bottom: 1px solid #f0f0f0; margin-bottom: 4px; background: #fff; }
        .meta-col { width: 45px; flex-shrink: 0; display: flex; flex-direction: column; align-items: center; margin-right: 6px; border-right: 1px solid #f0f0f0; }
        .act-val { color: #28a745; font-weight: bold; font-size: 0.7rem; }
        .seq-col { flex-grow: 1; font-family: 'Consolas', monospace; font-size: 10px; line-height: 1.1; white-space: pre-wrap; cursor: pointer; padding-left: 4px; }
        .token { display: inline; padding: 0 1px; border-radius: 2px; position: relative; }
        .token:hover { outline: 1px solid #444; z-index: 10; }
        .target-token { border-bottom: 2px solid #333; font-weight: bold; }
        .token .tooltip { visibility: hidden; background-color: #333; color: #fff; text-align: center; padding: 4px; border-radius: 4px; position: absolute; z-index: 100; bottom: 120%; left: 50%; transform: translateX(-50%); font-size: 10px; white-space: nowrap; pointer-events: none; opacity: 0; transition: opacity 0.2s; }
        .token:hover .tooltip { visibility: visible; opacity: 1; }
    </style>"""
    html_content = f"""<div class="activation-viz">{css_content}
        <div class="act-header" style="padding-bottom:2px; margin-bottom:4px; border-bottom:1px solid #eee;"><strong>Max Act:</strong> {data['max_act']:.4f}</div>"""
    max_global = data['max_act']
    for interval in data['intervals']:
        density_pct = interval['density'] * 100
        html_content += f"""<div class="interval-block"><div class="interval-header"><span>{interval['min']:.2f}-{interval['max']:.2f}</span><span>{density_pct:.5f}%</span></div>"""
        for ex in interval['examples']:
            short_html = _render_token_html(ex['short']['tokens'], ex['short']['activations'], ex['short']['target_idx'], max_global)
            full_html = _render_token_html(ex['full']['tokens'], ex['full']['activations'], ex['full']['target_idx'], max_global)
            html_content += f"""<div class="example-row"><div class="meta-col"><span class="act-val">{ex['target_act']:.2f}</span></div><div class="seq-col" onclick="this.querySelector('.short-view').style.display=this.querySelector('.short-view').style.display==='none'?'block':'none';this.querySelector('.full-view').style.display=this.querySelector('.full-view').style.display==='none'?'block':'none';"><div class="short-view">{short_html}</div><div class="full-view" style="display:none;">{full_html}</div></div></div>"""
        html_content += "</div>"
    html_content += "</div>"
    return html_content

# ---------------------------------------------------------
# 2. Main Logic
# ---------------------------------------------------------

class SAEGraphVisualizer:
    def __init__(
        self, 
        graph_obj, 
        model : HookedTransformer, 
        sae_dict: Dict[str, Any], 
        prompt_str, 
        data_toks: Tensor,
        batchsize: int = 8,
        full_data_dict: Dict[str, Tuple[Tensor, Tensor]] | None = None, 
        important_output_tokens: List[Tuple[int, int]] | None = None,
    ):
        self.graph = graph_obj
        self.model = model
        self.sae_dict = sae_dict
        self.prompt_str = prompt_str
        self.full_data_dict = full_data_dict
        self.data_toks = data_toks
        self.important_output_tokens = important_output_tokens
        self.batchsize = batchsize
        self.graph_json = None
        self.compressed_graph_json = None
        
        if hasattr(self.model, 'to_str_tokens'):
            self.tokens = self.model.to_str_tokens(self.prompt_str, prepend_bos=False)[0]
        else:
            self.tokens = self.prompt_str.split()
            
        self.node_clusters = {} # Store cluster assignments {node_id: cluster_label}
        
    def get_feature_acts(self) -> None:
        if self.full_data_dict is None:
            feature_dict = {}
            for (node_key, _), sparse_act in self.graph.nodes.items():
                hook = node_key.name
                if hook not in feature_dict:
                    feature_dict[hook] = set()
                
                if sparse_act.act is not None:
                    for s, f in zip(*sparse_act.act.nonzero(as_tuple=True)):
                        s_item, f_item = s.item(), f.item()
                        feature_dict[hook].add(f_item)
                        
            for key, val in feature_dict.items():
                feature_dict[key] = list(val)
                        
            self.full_data_dict = get_sae_activation_from_tok(
                self.sae_dict,
                self.model,
                self.data_toks,
                self.batchsize,
                feature_dict,
                True,
            )  
            
    @t.no_grad()
    def run_clustering(self, n_clusters=None, distance_threshold=1.5, n_intervals: int = 5, k_per_interval: int = 5, buffer: int = 10):
        """
        Runs clustering on feature nodes using Global Concatenated Network States.
        1. Extracts representation vectors by concatenating all layers' residual streams.
        2. Computes the centroid for each feature over the concatenated state.
        3. Clusters the concatenated vectors using AgglomerativeClustering.
        """
        temp_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        feature_nodes = [n for n in temp_data['nodes'] if n['type'] == 'feature']
        
        if not feature_nodes:
            print("No feature nodes found to cluster.")
            return

        print(f"Processing {len(feature_nodes)} nodes for global concat clustering...")
        
        hook_names = list(self.sae_dict.keys())
        saes = list(self.sae_dict.values())
        for sae in saes:
            sae.reset_hooks()
            
        # Define all layer hooks to concatenate
        ordered_model_hooks = ["blocks.{layer}.hook_resid_post".format(layer=layer) for layer in range(self.model.cfg.n_layers)]
        L_hooks = len(ordered_model_hooks)
        d_model = self.model.cfg.d_model
        concat_dim = L_hooks * d_model
        print(f"Concatenating representations across {L_hooks} hooks (dim={concat_dim})...")
        
        tensor_list = []
        valid_node_list = []
        valid_nodes_dict = {} 
        all_feats = {} 
        for node in feature_nodes:
            feat, hook = node['feat_idx'], node['hook']
            if hook not in self.sae_dict: continue
            
            if hook not in all_feats:
                all_feats[hook] = []
                valid_nodes_dict[hook] = []
            all_feats[hook].append(feat)
            valid_nodes_dict[hook].append(node)
            
        # Update accumulator dictionary to hold the concatenated dimension size
        accumulated_sum_dict = {
            hook: t.zeros(
                (len(all_feats[hook]), concat_dim), 
                device=self.model.cfg.device, 
                dtype=t.float32
            )
            for hook in all_feats.keys()
        }
        counts_dict = {
            hook: t.zeros((len(all_feats[hook])), device=self.model.cfg.device)
            for hook in all_feats.keys()
        }
        
        masks_dict = {}
                    
        def hook_sae_fn(inputs: Tensor, hook: Any, sae_name: str) -> None:
            if "acts_post" in hook.name:
                # Store the binary mask on the GPU
                masks_dict[sae_name] = (inputs.detach() > 1e-3).float()
                
        for sae in saes:
            sae.add_hook(
                lambda name: "acts_post" in name, partial(hook_sae_fn, sae_name=sae.cfg.hook_name), dir="fwd"
            )
            
        def hook_fn(tens: Tensor, hook: Any):
            sae = self.sae_dict[hook.name]
            sae(tens)

        dataloader = DataLoader(TensorDataset(self.data_toks), batch_size=8)
        
        for batch_toks in tqdm(dataloader, desc="Extracting Global Centroids"):
            toks = batch_toks[0].to(self.model.cfg.device)
            batch_cache = {}
            
            def make_cache_hook(h_name):
                def cache_hook_fn(tens: Tensor, hook: Any):
                    # Cache the full residual stream on the GPU
                    batch_cache[h_name] = tens.detach()
                return cache_hook_fn
                
            fwd_hooks = [(h_name, make_cache_hook(h_name)) for h_name in ordered_model_hooks]
            fwd_hooks.append((lambda name: name in hook_names, hook_fn))
            
            with self.model.hooks(fwd_hooks=fwd_hooks):
                self.model(toks)
                
            # Concatenate all layer states
            z_concat = t.cat([batch_cache[h] for h in ordered_model_hooks], dim=-1)
            z_flat = z_concat.view(-1, z_concat.size(-1)) # Shape: (batch * seq, L_hooks * d_model)
            
            for hook_name in all_feats.keys():
                if hook_name not in masks_dict: continue
                mask = masks_dict[hook_name]
                mask_flat = mask.view(-1, mask.size(-1)) # Shape: (batch * seq, d_sae)
                
                # Fast matrix multiplication: (d_sae, batch*seq) @ (batch*seq, concat_dim) -> (d_sae, concat_dim)
                # Cast to float32 BEFORE matmul to prevent float16 overflow
                accum_update = t.matmul(mask_flat.T.float(), z_flat.float())[all_feats[hook_name]]
                accumulated_sum_dict[hook_name] += accum_update
                
                counts_update = mask_flat.sum(dim=0)[all_feats[hook_name]]
                counts_dict[hook_name] += counts_update
                
            masks_dict.clear()
                    
        for sae in saes:
            sae.reset_hooks()

        for hook_name, accumulated_sum in accumulated_sum_dict.items():
            counts = counts_dict[hook_name]
            
            # Find indices where the feature actually activated
            valid_indices = (counts > 0).nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                continue
            
            valid_sums = accumulated_sum[valid_indices]
            valid_counts = counts[valid_indices]
            
            hook_nodes = valid_nodes_dict[hook_name]
            valid_hook_nodes = [hook_nodes[i.item()] for i in valid_indices]
            
            mean_vectors = valid_sums / valid_counts.unsqueeze(1)
            tensor_list.append(mean_vectors)
            valid_node_list.extend(valid_hook_nodes)

        # 3. Clustering
        if not tensor_list: 
            print("No features were activated across any hooks. Skipping clustering.")
            return
        
        data_matrix = t.cat(tensor_list, dim=0).float()
        X = data_matrix.detach().cpu().numpy()
        
        cluster_model = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(X)) if n_clusters is not None else None,
            metric="euclidean",
            distance_threshold=distance_threshold if n_clusters is None else None,
            linkage='ward'
        )
        print("Fitting clustering model...")
        labels = cluster_model.fit_predict(X)
        
        # 4. Store Results
        self.node_clusters = {n['id']: int(lbl) for n, lbl in zip(valid_node_list, labels)}
        for node in valid_node_list:
            for query_node in feature_nodes:
                if query_node['id'] == node['id']:
                    query_node['cluster'] = self.node_clusters[node['id']]
            
        print(f"Clustering complete. Found {len(set(labels))} clusters.")

    @t.no_grad()
    def run_clustering_ot_fast(
        self, 
        n_clusters=None, 
        distance_threshold=1.5, 
        max_points_per_feature=500, 
        alpha=0.5, 
        lambda_pen=1.0, 
        n_intervals: int = 5, 
        k_per_interval: int = 5, 
        buffer: int = 10
    ):
        """
        Runs clustering on feature nodes using:
        1. Semantic OT Distance (Projected to caching_hook)
        2. Functional Structural Equivalence (Edge connections)
        
        *OPTIMIZED*: Uses cached `self.full_data_dict` to surgically extract 
        only the required tokens, bypassing full-dataset inference.
        """
        temp_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        feature_nodes = [n for n in temp_data['nodes'] if n['type'] == 'feature']
        
        if not feature_nodes:
            print("No feature nodes found to cluster.")
            return

        print(f"Processing {len(feature_nodes)} nodes for clustering...")
        caching_hook = "ln_final.hook_normalized"
        
        all_feats = {} 
        valid_nodes_dict = {} 
        for node in feature_nodes:
            feat, hook = node['feat_idx'], node['hook']
            if hook not in self.sae_dict: continue
            
            if hook not in all_feats:
                all_feats[hook] = []
                valid_nodes_dict[hook] = []
            all_feats[hook].append(feat)
            valid_nodes_dict[hook].append(node)

        # =====================================================================
        # STEP 1: Query the Cache (Find Top Tokens Instantly)
        # =====================================================================
        print("Querying cached full_data_dict for top activations...")
        B, S = self.data_toks.shape
        feature_activations = {node['id']: {'b': [], 's': [], 'weights': []} for node in feature_nodes}

        for hook_name, feat_list in all_feats.items():
            if hook_name not in getattr(self, "full_data_dict", {}): 
                continue
                
            vals, idxs = self.full_data_dict[hook_name] # (B, S, K) # type: ignore
            
            # Move to CPU to slice memory efficiently
            vals = vals.cpu()
            idxs = idxs.cpu()
            
            # Global valid mask
            valid_mask = vals > 1e-3
            valid_vals = vals[valid_mask]
            valid_idxs = idxs[valid_mask]
            
            # Generate coordinates matching the mask
            K = vals.shape[-1]
            b_coords = t.arange(B).view(B, 1, 1).expand(B, S, K)[valid_mask]
            s_coords = t.arange(S).view(1, S, 1).expand(B, S, K)[valid_mask]
            
            for i, feat_idx in enumerate(feat_list):
                node_id = valid_nodes_dict[hook_name][i]['id']
                
                # Find exact occurrences of this feature
                f_mask = valid_idxs == feat_idx
                
                f_vals = valid_vals[f_mask]
                f_b = b_coords[f_mask]
                f_s = s_coords[f_mask]
                
                # Apply max_points restriction instantly (Global Top-K Sort)
                if len(f_vals) > max_points_per_feature:
                    top_vals, top_idx = t.topk(f_vals, max_points_per_feature)
                    f_vals = top_vals
                    f_b = f_b[top_idx]
                    f_s = f_s[top_idx]
                    
                feature_activations[node_id]['weights'] = f_vals # type: ignore
                feature_activations[node_id]['b'] = f_b # type: ignore
                feature_activations[node_id]['s'] = f_s # type: ignore

        # =====================================================================
        # STEP 2: Determine Required Subset & Coordinate Mapping
        # =====================================================================
        required_b_s_pairs = set()
        for data in feature_activations.values():
            for b, s in zip(data['b'], data['s']):
                required_b_s_pairs.add((b.item(), s.item()))
                
        required_batches = sorted(list(set([b for b, s in required_b_s_pairs])))
        num_needed_tokens = len(required_b_s_pairs)
        
        if num_needed_tokens == 0:
            print("No valid activations found in cache.")
            return

        # Map (batch, seq) directly to a row index in our dense latents tensor
        bs_to_row = {}
        row_idx = 0
        for orig_b in required_batches:
            for s in range(S):
                if (orig_b, s) in required_b_s_pairs:
                    bs_to_row[(orig_b, s)] = row_idx
                    row_idx += 1

        d_model = self.model.cfg.d_model
        global_latents = t.zeros((num_needed_tokens, d_model), dtype=t.float16, device='cpu')

        # =====================================================================
        # STEP 3: Surgical Forward Pass (Extract caching_hook latents only)
        # =====================================================================
        print(f"Executing surgical extraction on {len(required_batches)} required batches...")
        
        for i in tqdm(range(0, len(required_batches), self.batchsize), desc="Extracting Latents"):
            batch_b_indices = required_batches[i : i+self.batchsize]
            toks = self.data_toks[batch_b_indices].to(self.model.cfg.device)
            
            batch_cache = {}
            def cache_hook_fn(tens: Tensor, hook: Any):
                # Shape: (subset_batch, S, d_model)
                batch_cache['acts'] = tens.detach().half()
                
            # Execute model (No SAE hooks needed!)
            with self.model.hooks(fwd_hooks=[(caching_hook, cache_hook_fn)]):
                self.model(toks)
                
            z_acts = batch_cache['acts']
            
            # Slot the specific tokens into the global flat tensor
            for local_b, orig_b in enumerate(batch_b_indices):
                for s in range(S):
                    if (orig_b, s) in required_b_s_pairs:
                        row = bs_to_row[(orig_b, s)]
                        global_latents[row] = z_acts[local_b, s].cpu()

        # =====================================================================
        # STEP 4: Build OT Processed Sets
        # =====================================================================
        valid_node_list = []
        processed_features = []
        
        for node in feature_nodes:
            node_id = node['id']
            f_b = feature_activations[node_id]['b']
            f_s = feature_activations[node_id]['s']
            W = feature_activations[node_id]['weights']
            
            if len(W) == 0: continue
                
            row_indices = [bs_to_row[(b.item(), s.item())] for b, s in zip(f_b, f_s)]
            X = global_latents[row_indices].float() # Cast back up to float32 for math precision
            
            W_np = W.detach().cpu().float().numpy().astype(np.float64) # type: ignore
            W_np = W_np / W_np.sum()
            W_np[-1] = 1.0 - np.sum(W_np[:-1])

            processed_features.append({'X': X, 'W': W_np})
            valid_node_list.append(node)
            
        n_valid = len(valid_node_list)
        if n_valid < 2: 
            print("Not enough activated features to cluster. Skipping.")
            return

        # =====================================================================
        # STEP 5: Compute Semantic OT Distance Matrix (D_sem)
        # =====================================================================
        print(f"Computing Optimal Transport distance matrix for {n_valid} features...")
        D_sem = np.zeros((n_valid, n_valid))
        device = self.model.cfg.device
        
        for i in tqdm(range(n_valid), desc="OT Distances"):
            # Send target feature latents to GPU once
            X_i_t = processed_features[i]['X'].to(device)
            w_i = processed_features[i]['W']
            
            for j in range(i + 1, n_valid):
                # Send comparison feature latents to GPU
                X_j_t = processed_features[j]['X'].to(device)
                w_j = processed_features[j]['W']
                
                # FAST PYTORCH CDIST
                cost_matrix_t = t.cdist(X_i_t, X_j_t, p=2)
                cost_matrix_np = cost_matrix_t.cpu().numpy().astype(np.float64)
                
                ot_dist = ot.emd2(w_i, w_j, cost_matrix_np)
                D_sem[i, j] = ot_dist
                D_sem[j, i] = ot_dist 

        print(f"Optimal Transport distance matrix: min={D_sem.min():.4f}, max={D_sem.max():.4f}, mean={D_sem.mean():.4f}")
        
        # =====================================================================
        # STEP 6: Compute Functional Wiring Distance Matrix (D_func)
        # =====================================================================
        print("Computing Functional Wiring Distance matrix...")
        
        all_nodes = temp_data['nodes']
        N_all = len(all_nodes)
        node_id_to_idx = {n['id']: i for i, n in enumerate(all_nodes)}
        
        W_adj = np.zeros((N_all, N_all), dtype=np.float32)
        
        for link in temp_data['links']:
            src = node_id_to_idx.get(link['source'])
            tgt = node_id_to_idx.get(link['target'])
            if src is not None and tgt is not None:
                W_adj[src, tgt] = link['value']
                
        max_w = np.max(np.abs(W_adj))
        if max_w > 0:
            W_adj = W_adj / max_w
        np.fill_diagonal(W_adj, 1.0)
        
        valid_indices = [node_id_to_idx[n['id']] for n in valid_node_list]
        V_signatures = np.zeros((n_valid, 2 * N_all), dtype=np.float32)
        
        for i, u in enumerate(valid_indices):
            V_signatures[i, :N_all] = W_adj[u, :] 
            V_signatures[i, N_all:] = W_adj[:, u] 
            
        D_func = squareform(pdist(V_signatures, metric='cosine'))
        
        print(f"Base Functional Wiring distance matrix: min={D_func.min():.4f}, max={D_func.max():.4f}, mean={D_func.mean():.4f}")
        
        D_func_penalized = np.copy(D_func)
        for i in range(n_valid):
            u = valid_indices[i]
            for j in range(i + 1, n_valid):
                v = valid_indices[j]
                penalty = lambda_pen * max(0, -W_adj[u, v] - W_adj[v, u])
                D_func_penalized[i, j] += penalty
                D_func_penalized[j, i] += penalty
                
        print(f"Penalized Functional Wiring distance matrix: min={D_func_penalized.min():.4f}, max={D_func_penalized.max():.4f}, mean={D_func_penalized.mean():.4f}")

        # =====================================================================
        # STEP 7: Combine and Cluster (D_total)
        # =====================================================================
        print(f"Blending Semantic and Functional distances (alpha={alpha})...")
        
        max_sem = np.max(D_sem)
        D_sem_norm = D_sem / max_sem if max_sem > 0 else D_sem
        
        max_func = np.max(D_func_penalized)
        D_func_norm = D_func_penalized / max_func if max_func > 0 else D_func_penalized
        
        D_total = alpha * D_sem_norm + (1 - alpha) * D_func_norm

        print("Fitting agglomerative clustering model...")
        cluster_model = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(D_total)) if n_clusters is not None else None,
            metric="precomputed", 
            distance_threshold=distance_threshold if n_clusters is None else None,
            linkage='complete' 
        )
        
        labels = cluster_model.fit_predict(D_total)
        
        # =====================================================================
        # STEP 8: Store Results
        # =====================================================================
        self.node_clusters = {n['id']: int(lbl) for n, lbl in zip(valid_node_list, labels)}
        for node in valid_node_list:
            for query_node in feature_nodes:
                if query_node['id'] == node['id']:
                    query_node['cluster'] = self.node_clusters[node['id']]
            
        print(f"Clustering complete. Found {len(set(labels))} clusters.")
    

    @t.no_grad()
    def run_clustering_ot_global_concat_fast(
        self, 
        n_clusters=None, 
        distance_threshold=1.5, 
        max_points_per_feature=500, 
        alpha=0.5, 
        lambda_pen=1.0, 
        n_intervals: int = 5, 
        k_per_interval: int = 5, 
        buffer: int = 10
    ):
        """
        Runs clustering on feature nodes using Global Concatenated Network States.
        - Optimized: Uses `self.full_data_dict` for surgical token extraction.
        - Optimized: Uses PyTorch `t.cdist` on GPU for cost matrix computation.
        """
        temp_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        feature_nodes = [n for n in temp_data['nodes'] if n['type'] == 'feature']
        
        if not feature_nodes:
            print("No feature nodes found to cluster.")
            return

        print(f"Processing {len(feature_nodes)} nodes for clustering...")
        device = self.model.cfg.device
        
        saes = list(self.sae_dict.values())
        # ordered_model_hooks = list(dict.fromkeys([sae.cfg.hook_name for sae in saes]))
        ordered_model_hooks = ["blocks.{layer}.hook_resid_post".format(layer=layer) for layer in range(self.model.cfg.n_layers)]
        print(f"Concatenating representations across {len(ordered_model_hooks)} hooks...")
        
        all_feats = {} 
        valid_nodes_dict = {} 
        for node in feature_nodes:
            feat, hook = node['feat_idx'], node['hook']
            if hook not in self.sae_dict: continue
            
            if hook not in all_feats:
                all_feats[hook] = []
                valid_nodes_dict[hook] = []
            all_feats[hook].append(feat)
            valid_nodes_dict[hook].append(node)

        # =====================================================================
        # STEP 1: Query the Cache (Find Top Tokens Instantly)
        # =====================================================================
        print("Querying cached full_data_dict for top activations...")
        B, S = self.data_toks.shape
        feature_activations = {node['id']: {'b': [], 's': [], 'weights': []} for node in feature_nodes}

        for hook_name, feat_list in all_feats.items():
            if hook_name not in getattr(self, "full_data_dict", {}): continue
                
            vals, idxs = self.full_data_dict[hook_name] # (B, S, K) # type: ignore
            vals, idxs = vals.cpu(), idxs.cpu()
            
            valid_mask = vals > 1e-3
            valid_vals, valid_idxs = vals[valid_mask], idxs[valid_mask]
            
            K = vals.shape[-1]
            b_coords = t.arange(B).view(B, 1, 1).expand(B, S, K)[valid_mask]
            s_coords = t.arange(S).view(1, S, 1).expand(B, S, K)[valid_mask]
            
            for i, feat_idx in enumerate(feat_list):
                node_id = valid_nodes_dict[hook_name][i]['id']
                f_mask = valid_idxs == feat_idx
                
                f_vals = valid_vals[f_mask]
                f_b, f_s = b_coords[f_mask], s_coords[f_mask]
                
                if len(f_vals) > max_points_per_feature:
                    top_vals, top_idx = t.topk(f_vals, max_points_per_feature)
                    f_vals, f_b, f_s = top_vals, f_b[top_idx], f_s[top_idx]
                    
                feature_activations[node_id]['weights'] = f_vals # type: ignore
                feature_activations[node_id]['b'] = f_b # type: ignore
                feature_activations[node_id]['s'] = f_s # type: ignore

        # =====================================================================
        # STEP 2: Determine Required Subset & Coordinate Mapping
        # =====================================================================
        required_b_s_pairs = set()
        for data in feature_activations.values():
            for b, s in zip(data['b'], data['s']):
                required_b_s_pairs.add((b.item(), s.item()))
                
        required_batches = sorted(list(set([b for b, s in required_b_s_pairs])))
        num_needed_tokens = len(required_b_s_pairs)
        
        if num_needed_tokens == 0:
            print("No valid activations found in cache.")
            return

        bs_to_row = {}
        row_idx = 0
        for orig_b in required_batches:
            for s in range(S):
                if (orig_b, s) in required_b_s_pairs:
                    bs_to_row[(orig_b, s)] = row_idx
                    row_idx += 1

        L_hooks = len(ordered_model_hooks)
        d_model = self.model.cfg.d_model
        global_latents = t.zeros((num_needed_tokens, L_hooks * d_model), dtype=t.float16, device='cpu')

        # =====================================================================
        # STEP 3: Surgical Forward Pass (Concatenated States)
        # =====================================================================
        print(f"Executing surgical extraction on {len(required_batches)} required batches...")
        
        for i in tqdm(range(0, len(required_batches), self.batchsize), desc="Extracting Global States"):
            batch_b_indices = required_batches[i : i+self.batchsize]
            toks = self.data_toks[batch_b_indices].to(device)
            
            batch_cache = {}
            def make_cache_hook(h_name):
                def cache_hook_fn(tens: Tensor, hook: Any):
                    batch_cache[h_name] = tens.detach().half()
                return cache_hook_fn
                
            fwd_hooks = [(h_name, make_cache_hook(h_name)) for h_name in ordered_model_hooks]
            
            with self.model.hooks(fwd_hooks=fwd_hooks): # type: ignore
                self.model(toks)
                
            z_concat = t.cat([batch_cache[h] for h in ordered_model_hooks], dim=-1)
            
            for local_b, orig_b in enumerate(batch_b_indices):
                for s in range(S):
                    if (orig_b, s) in required_b_s_pairs:
                        row = bs_to_row[(orig_b, s)]
                        global_latents[row] = z_concat[local_b, s].cpu()

        # =====================================================================
        # STEP 4: Build OT Processed Sets (Kept in CPU RAM as Float32)
        # =====================================================================
        valid_node_list = []
        processed_features = []
        
        for node in feature_nodes:
            node_id = node['id']
            f_b = feature_activations[node_id]['b']
            f_s = feature_activations[node_id]['s']
            W = feature_activations[node_id]['weights']
            
            if len(W) == 0: continue
                
            row_indices = [bs_to_row[(b.item(), s.item())] for b, s in zip(f_b, f_s)]
            X = global_latents[row_indices].float() # Float32 for cdist accuracy
            
            W_np = W.detach().cpu().float().numpy().astype(np.float64) # type: ignore
            W_np = W_np / W_np.sum()
            W_np[-1] = 1.0 - np.sum(W_np[:-1])

            processed_features.append({'X': X, 'W': W_np})
            valid_node_list.append(node)
            
        n_valid = len(valid_node_list)
        if n_valid < 2: 
            print("Not enough activated features to cluster. Skipping.")
            return

        # =====================================================================
        # STEP 5: Compute Semantic OT Distance Matrix (PyTorch GPU cdist)
        # =====================================================================
        print(f"Computing Optimal Transport distance matrix for {n_valid} features using PyTorch...")
        D_sem = np.zeros((n_valid, n_valid))
        
        for i in tqdm(range(n_valid), desc="OT Distances"):
            # Send target feature latents to GPU once
            X_i_t = processed_features[i]['X'].to(device)
            w_i = processed_features[i]['W']
            
            for j in range(i + 1, n_valid):
                # Send comparison feature latents to GPU
                X_j_t = processed_features[j]['X'].to(device)
                w_j = processed_features[j]['W']
                
                # FAST PYTORCH CDIST
                cost_matrix_t = t.cdist(X_i_t, X_j_t, p=2)
                
                # Download tiny distance matrix back to CPU for POT solver
                cost_matrix_np = cost_matrix_t.cpu().numpy().astype(np.float64)
                ot_dist = ot.emd2(w_i, w_j, cost_matrix_np)
                
                D_sem[i, j] = ot_dist
                D_sem[j, i] = ot_dist 

        print(f"Optimal Transport distance matrix: min={D_sem.min():.4f}, max={D_sem.max():.4f}, mean={D_sem.mean():.4f}")
        
        # =====================================================================
        # STEP 6: Compute Functional Wiring Distance Matrix (D_func)
        # =====================================================================
        print("Computing Functional Wiring Distance matrix...")
        all_nodes = temp_data['nodes']
        N_all = len(all_nodes)
        node_id_to_idx = {n['id']: i for i, n in enumerate(all_nodes)}
        
        W_adj = np.zeros((N_all, N_all), dtype=np.float32)
        for link in temp_data['links']:
            src = node_id_to_idx.get(link['source'])
            tgt = node_id_to_idx.get(link['target'])
            if src is not None and tgt is not None:
                W_adj[src, tgt] = link['value']
                
        max_w = np.max(np.abs(W_adj))
        if max_w > 0: W_adj = W_adj / max_w
        np.fill_diagonal(W_adj, 1.0)
        
        valid_indices = [node_id_to_idx[n['id']] for n in valid_node_list]
        V_signatures = np.zeros((n_valid, 2 * N_all), dtype=np.float32)
        
        for i, u in enumerate(valid_indices):
            V_signatures[i, :N_all] = W_adj[u, :] 
            V_signatures[i, N_all:] = W_adj[:, u] 
            
        D_func = squareform(pdist(V_signatures, metric='cosine'))
        
        D_func_penalized = np.copy(D_func)
        for i in range(n_valid):
            u = valid_indices[i]
            for j in range(i + 1, n_valid):
                v = valid_indices[j]
                penalty = lambda_pen * max(0, -W_adj[u, v] - W_adj[v, u])
                D_func_penalized[i, j] += penalty
                D_func_penalized[j, i] += penalty

        # =====================================================================
        # STEP 7: Combine and Cluster 
        # =====================================================================
        print(f"Blending Semantic and Functional distances (alpha={alpha})...")
        
        max_sem = np.max(D_sem)
        D_sem_norm = D_sem / max_sem if max_sem > 0 else D_sem
        
        max_func = np.max(D_func_penalized)
        D_func_norm = D_func_penalized / max_func if max_func > 0 else D_func_penalized
        
        D_total = alpha * D_sem_norm + (1 - alpha) * D_func_norm

        print("Fitting agglomerative clustering model...")
        cluster_model = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(D_total)) if n_clusters is not None else None,
            metric="precomputed", 
            distance_threshold=distance_threshold if n_clusters is None else None,
            linkage='complete' 
        )
        
        labels = cluster_model.fit_predict(D_total)
        
        self.node_clusters = {n['id']: int(lbl) for n, lbl in zip(valid_node_list, labels)}
        for node in valid_node_list:
            for query_node in feature_nodes:
                if query_node['id'] == node['id']:
                    query_node['cluster'] = self.node_clusters[node['id']]
            
        print(f"Clustering complete. Found {len(set(labels))} clusters.")
        
    @t.no_grad()
    def run_clustering_causal(
        self, 
        n_clusters=None, 
        distance_threshold=1.5, 
        lambda_pen=1.0, 
        n_intervals: int = 5, 
        k_per_interval: int = 5, 
        buffer: int = 10
    ):
        """
        Runs clustering on feature nodes using:
        1. Semantic OT Distance (Projected to caching_hook)
        2. Functional Structural Equivalence (Edge connections)
        
        *OPTIMIZED*: Uses cached `self.full_data_dict` to surgically extract 
        only the required tokens, bypassing full-dataset inference.
        """
        temp_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        feature_nodes = [n for n in temp_data['nodes'] if n['type'] == 'feature']
        
        if not feature_nodes:
            print("No feature nodes found to cluster.")
            return

        print(f"Processing {len(feature_nodes)} nodes for clustering...")
        
        saes = list(self.sae_dict.values())
        for sae in saes:
            sae.reset_hooks()
        
        valid_node_list = []
        valid_nodes_dict = {} 
        all_feats = {hook: [] for hook in self.sae_dict} 

        for node in feature_nodes:
            feat, hook = node['feat_idx'], node['hook']
            if hook not in self.sae_dict: continue
            
            if hook not in valid_nodes_dict:
                valid_nodes_dict[hook] = []
            all_feats[hook].append(feat)
            valid_nodes_dict[hook].append(node)
            
        # Trackers for filtering and folding
        counts_dict = {
            hook: t.zeros(len(all_feats[hook]), device=self.model.cfg.device)
            for hook in all_feats.keys()
        }

        # 2. Lightweight Forward Pass (Only to find alive features)
        def hook_sae_fn(inputs: Tensor, hook: Any, sae_name: str) -> None:
            if "acts_post" in hook.name:
                acts = inputs.detach().view(-1, inputs.shape[-1])
                
                # Slice only the features we care about for this SAE
                feat_indices = all_feats[sae_name]
                f_acts = acts[:, feat_indices]
                f_mask = f_acts > 1e-3
                
                counts_dict[sae_name] += f_mask.sum(dim=0)
                
        for sae in saes:
            sae.add_hook(
                lambda name: "acts_post" in name, 
                partial(hook_sae_fn, sae_name=sae.cfg.hook_name), 
                dir="fwd"
            )
            
        def hook_fn(tens: Tensor, hook: Any):
            sae = self.sae_dict[hook.name]
            sae(tens)

        dataloader = DataLoader(TensorDataset(self.data_toks), batch_size=8)
        with self.model.hooks(fwd_hooks=[(lambda name: name in self.sae_dict.keys(), hook_fn)]):
            for toks in tqdm(dataloader, desc="Extracting Feature Stats"):
                self.model(toks[0])
                    
        for sae in saes:
            sae.reset_hooks()
        
        for hook_name, counts in counts_dict.items():
            # Find indices where the feature actually activated > 0 times
            valid_local_indices = (counts > 0).nonzero(as_tuple=True)[0]
            
            if len(valid_local_indices) == 0:
                continue
            
            hook_nodes = valid_nodes_dict[hook_name]
            valid_hook_nodes = [hook_nodes[i.item()] for i in valid_local_indices]
            valid_node_list.extend(valid_hook_nodes)
            
        n_valid = len(valid_node_list)
        if n_valid < 2: 
            print("Not enough activated features to cluster. Skipping.")
            return
        
        # =====================================================================
        # STEP 6: Compute Functional Wiring Distance Matrix (D_func)
        # =====================================================================
        print("Computing Functional Wiring Distance matrix...")
        
        all_nodes = temp_data['nodes']
        N_all = len(all_nodes)
        node_id_to_idx = {n['id']: i for i, n in enumerate(all_nodes)}
        
        W_adj = np.zeros((N_all, N_all), dtype=np.float32)
        
        for link in temp_data['links']:
            src = node_id_to_idx.get(link['source'])
            tgt = node_id_to_idx.get(link['target'])
            if src is not None and tgt is not None:
                W_adj[src, tgt] = link['value']
                
        max_w = np.max(np.abs(W_adj))
        if max_w > 0:
            W_adj = W_adj / max_w
        np.fill_diagonal(W_adj, 1.0)
        
        valid_indices = [node_id_to_idx[n['id']] for n in valid_node_list]
        V_signatures = np.zeros((n_valid, 2 * N_all), dtype=np.float32)
        
        for i, u in enumerate(valid_indices):
            V_signatures[i, :N_all] = W_adj[u, :] 
            V_signatures[i, N_all:] = W_adj[:, u] 
            
        D_func = squareform(pdist(V_signatures, metric='cosine'))
        
        print(f"Base Functional Wiring distance matrix: min={D_func.min():.4f}, max={D_func.max():.4f}, mean={D_func.mean():.4f}")
        
        D_func_penalized = np.copy(D_func)
        for i in range(n_valid):
            u = valid_indices[i]
            for j in range(i + 1, n_valid):
                v = valid_indices[j]
                penalty = lambda_pen * max(0, -W_adj[u, v] - W_adj[v, u])
                D_func_penalized[i, j] += penalty
                D_func_penalized[j, i] += penalty
                
        print(f"Penalized Functional Wiring distance matrix: min={D_func_penalized.min():.4f}, max={D_func_penalized.max():.4f}, mean={D_func_penalized.mean():.4f}")
        
        max_func = np.max(D_func_penalized)
        D_func_norm = D_func_penalized / max_func if max_func > 0 else D_func_penalized

        print("Fitting agglomerative clustering model...")
        cluster_model = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(D_func_norm)) if n_clusters is not None else None,
            metric="precomputed", 
            distance_threshold=distance_threshold if n_clusters is None else None,
            linkage='complete' 
        )
        
        labels = cluster_model.fit_predict(D_func_norm)
        
        # =====================================================================
        # STEP 8: Store Results
        # =====================================================================
        self.node_clusters = {n['id']: int(lbl) for n, lbl in zip(valid_node_list, labels)}
        for node in valid_node_list:
            for query_node in feature_nodes:
                if query_node['id'] == node['id']:
                    query_node['cluster'] = self.node_clusters[node['id']]
            
        print(f"Clustering complete. Found {len(set(labels))} clusters.")
    
    @t.no_grad()
    def run_clustering_decoder(
        self, 
        method: str = "feature_flow", # 'feature_flow' or 'mech_permute'
        n_clusters=None, 
        distance_threshold=1.5, 
        n_intervals: int = 5, 
        k_per_interval: int = 5, 
        buffer: int = 10
    ):
        """
        Runs clustering on feature nodes using SAE decoder weights (W_dec).
        - 'feature_flow': Clusters using Cosine Similarity of normalized W_dec.
        - 'mech_permute': Clusters using Euclidean (L2) distance of Folded W_dec.
        """
        assert method in ["feature_flow", "mech_permute"], "Method must be 'feature_flow' or 'mech_permute'"

        # 1. Build list of Feature Nodes (exclude errors)
        temp_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        feature_nodes = [n for n in temp_data['nodes'] if n['type'] == 'feature']
        
        if not feature_nodes:
            print("No feature nodes found to cluster.")
            return

        print(f"Processing {len(feature_nodes)} nodes for clustering using {method.upper()}...")
        
        saes = list(self.sae_dict.values())
        for sae in saes:
            sae.reset_hooks()
        
        valid_node_list = []
        valid_nodes_dict = {} 
        all_feats = {hook: [] for hook in self.sae_dict} 

        for node in feature_nodes:
            feat, hook = node['feat_idx'], node['hook']
            if hook not in self.sae_dict: continue
            
            if hook not in valid_nodes_dict:
                valid_nodes_dict[hook] = []
            all_feats[hook].append(feat)
            valid_nodes_dict[hook].append(node)
            
        # Trackers for filtering and folding
        counts_dict = {
            hook: t.zeros(len(all_feats[hook]), device=self.model.cfg.device)
            for hook in all_feats.keys()
        }
        
        min_acts_dict = {
            hook: t.full((len(all_feats[hook]),), float('inf'), device=self.model.cfg.device)
            for hook in all_feats.keys()
        }

        # 2. Lightweight Forward Pass (Only to find alive features & min activations)
        def hook_sae_fn(inputs: Tensor, hook: Any, sae_name: str) -> None:
            if "acts_post" in hook.name:
                acts = inputs.detach().view(-1, inputs.shape[-1])
                
                # Slice only the features we care about for this SAE
                feat_indices = all_feats[sae_name]
                f_acts = acts[:, feat_indices]
                f_mask = f_acts > 1e-3
                
                counts_dict[sae_name] += f_mask.sum(dim=0)
                
                if method == "mech_permute":
                    # Find the minimum non-zero activation (proxy for JumpReLU theta)
                    active_only = t.where(f_mask, f_acts, float('inf'))
                    batch_min = active_only.amin(dim=0)
                    min_acts_dict[sae_name] = t.minimum(min_acts_dict[sae_name], batch_min)
                
        for sae in saes:
            sae.add_hook(
                lambda name: "acts_post" in name, 
                partial(hook_sae_fn, sae_name=sae.cfg.hook_name), 
                dir="fwd"
            )
            
        def hook_fn(tens: Tensor, hook: Any):
            sae = self.sae_dict[hook.name]
            sae(tens)

        dataloader = DataLoader(TensorDataset(self.data_toks), batch_size=8)
        with self.model.hooks(fwd_hooks=[(lambda name: name in self.sae_dict.keys(), hook_fn)]):
            for toks in tqdm(dataloader, desc="Extracting Feature Stats"):
                self.model(toks[0])
                    
        for sae in saes:
            sae.reset_hooks()

        # 3. Build Feature Matrix (W_dec extraction)
        tensor_list = []
        
        for hook_name, counts in counts_dict.items():
            # Find indices where the feature actually activated > 0 times
            valid_local_indices = (counts > 0).nonzero(as_tuple=True)[0]
            
            if len(valid_local_indices) == 0:
                continue
                
            # Map local valid index back to the global feature index in the SAE
            actual_feat_indices = [all_feats[hook_name][i.item()] for i in valid_local_indices] # type: ignore
            
            sae = self.sae_dict[hook_name]
            W_dec_valid = sae.W_dec[actual_feat_indices].to(self.model.cfg.device).float()
            
            if method == "mech_permute":
                # Fold parameters: W_dec * diag(theta)
                min_acts = min_acts_dict[hook_name][valid_local_indices]
                
                # Replace remaining 'inf' (if a feature somehow slipped through) with 0.0
                clean_min_acts = t.where(t.isinf(min_acts), t.zeros_like(min_acts), min_acts)
                W_processed = W_dec_valid * clean_min_acts.unsqueeze(1)
                
            elif method == "feature_flow":
                # Feature Flow normalizes the decoder weights for Cosine Similarity
                W_processed = t.nn.functional.normalize(W_dec_valid, p=2, dim=1)
                
            tensor_list.append(W_processed)
            
            hook_nodes = valid_nodes_dict[hook_name]
            valid_hook_nodes = [hook_nodes[i.item()] for i in valid_local_indices]
            valid_node_list.extend(valid_hook_nodes)

        if not tensor_list: 
            print("No features were activated across any hooks. Skipping clustering.")
            return
        
        data_matrix = t.cat(tensor_list, dim=0)
        X = data_matrix.detach().cpu().numpy()
        
        # 4. Clustering
        print("Fitting clustering model...")
        
        # Sklearn linkage constraints: 'ward' only works with 'euclidean'. 
        # For 'cosine', we must use 'average' or 'complete'.
        cluster_metric = "euclidean" if method == "mech_permute" else "cosine"
        cluster_linkage = "ward" if method == "mech_permute" else "average"
        
        cluster_model = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(X)) if n_clusters is not None else None,
            metric=cluster_metric,
            distance_threshold=distance_threshold if n_clusters is None else None,
            linkage=cluster_linkage
        )
        
        labels = cluster_model.fit_predict(X)
        
        # 5. Store Results
        self.node_clusters = {n['id']: int(lbl) for n, lbl in zip(valid_node_list, labels)}
        for node in valid_node_list:
            for query_node in feature_nodes:
                if query_node['id'] == node['id']:
                    query_node['cluster'] = self.node_clusters[node['id']]
            
        print(f"Clustering complete. Found {len(set(labels))} clusters.")

    @t.no_grad()
    def run_clustering_llm(
        self, 
        n_clusters: int,   
        n_intervals: int = 5, 
        k_per_interval: int = 5, 
        buffer: int = 10,
        fallback_method: str = "feature_flow" 
    ):
        """
        Runs clustering on feature nodes by asking an LLM to interpret and group them based on 
        semantics. Enforces an exact number of clusters if `n_clusters` is provided.
        """
        temp_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        feature_nodes = [n for n in temp_data['nodes'] if n['type'] == 'feature']
        
        if not feature_nodes:
            print("No feature nodes found to cluster.")
            return

        print(f"Processing {len(feature_nodes)} nodes for LLM clustering...")
        
        # 1. Format the data and generate the dynamic prompt
        formatted_features_text = format_features_for_llm_clustering(temp_data)
        system_prompt = get_llm_clustering_prompt(n_clusters)
        
        # 2. Call the LLM (Wrapped in try/except for Fallback)
        llm_success = False
        parsed_json = None
        
        try:
            print(f"Prompting LLM for semantic clustering. (Target clusters: {n_clusters if n_clusters else 'Auto'})")
            # Key for the LLM Sel endpoint (see call_llm_for_clustering): LLMSEL_API_KEY, else the
            # UF gateway's UF_AI_API_KEY, else OPENAI_API_KEY.
            uf_api_key = os.getenv("LLMSEL_API_KEY") or os.getenv("UF_AI_API_KEY") or API_KEY
            raw_response = call_llm_for_clustering(uf_api_key, system_prompt, formatted_features_text) # type: ignore
            
            # Clean up potential markdown blocks
            json_str = re.sub(r"```json\s*", "", raw_response)
            json_str = re.sub(r"\s*```", "", json_str)
            
            parsed_json = json.loads(json_str)
            
            # Validation Checks
            if "clusters" not in parsed_json or not isinstance(parsed_json["clusters"], list):
                raise ValueError("LLM response did not contain the 'clusters' array.")
            
            returned_cluster_count = len(parsed_json["clusters"])
            if n_clusters is not None and returned_cluster_count != n_clusters:
                print(f"[NOTE] Requested {n_clusters} clusters, but LLM returned {returned_cluster_count}. Forcing assignment anyway.")
                
            llm_success = True
            
        except Exception as e:
            print(f"\n[WARNING] LLM Clustering failed: {str(e)}")
            print(f"Falling back to geometric/activation clustering method: '{fallback_method}'...\n")
        
        # 3. Apply the results OR trigger the fallback
        if llm_success and parsed_json is not None:
            print("LLM clustering successful. Assigning clusters...")
            
            feature_to_cluster = {}
            for cluster_data in parsed_json["clusters"]:
                c_id = int(cluster_data.get("cluster_id", 0))
                for f_id in cluster_data.get("feature_ids", []):
                    feature_to_cluster[f_id] = c_id
            
            
            self.node_clusters = {}
            for node in feature_nodes:
                node_id = node['id']
                cluster_id = None
                if node_id in feature_to_cluster:
                    # check if valid cluster assignment
                    if (feature_to_cluster[node_id] >= 0) or (feature_to_cluster[node_id] < n_clusters):
                        cluster_id = feature_to_cluster[node_id]

                if cluster_id is None: # else, assign randomly
                    print(f"Warning: LLM missed assigned feature {node_id}.")
                    cluster_id = random.randint(0, n_clusters - 1)
                
                self.node_clusters[node_id] = cluster_id
                
                # Update the node dicts inplace
                node['cluster'] = cluster_id
                node['llm_caption'] = next((c.get("cluster_caption", "No caption") for c in parsed_json["clusters"] if c.get("cluster_id") == cluster_id), "Misc")

            num_clusters = len(set(self.node_clusters.values()))
            print(f"LLM Clustering complete. Found {num_clusters} semantic clusters.")
            
        else:
            # ==========================================
            # FALLBACK MECHANISM
            # ==========================================
            if fallback_method == "feature_flow":
                self.run_clustering_decoder(method="feature_flow", n_clusters=n_clusters, n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
            elif fallback_method == "ot":
                self.run_clustering_ot_global_concat_fast(n_clusters=n_clusters, n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
            else:
                print(f"Unknown fallback method '{fallback_method}'. Features will remain unclustered.")
    
    def evaluate_clustering(self, epsilon=1e-9):
        """
        Calculates the Intra-Cluster Synergy (S_intra) and 
        Inter-Cluster Homogeneity (S_inter) for the current clustering.
        """
        graph_data = self.build_graph_json()
        
        # 1. Map nodes to clusters and filter valid clustered nodes
        node_to_cluster = {}
        for n in graph_data['nodes']:
            if n.get('cluster') is not None:
                node_to_cluster[n['id']] = n['cluster']
                
        if not node_to_cluster:
            print("No clustered nodes found. Run clustering first.")
            return None
            
        # Map valid nodes to integer indices for fast numpy operations
        valid_nodes = list(node_to_cluster.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(valid_nodes)}
        N = len(valid_nodes)
        
        # 2. Build the Adjacency Matrix (W) for clustered nodes
        W = np.zeros((N, N), dtype=np.float32)
        for link in graph_data['links']:
            src = link['source']
            tgt = link['target']
            if src in node_to_idx and tgt in node_to_idx:
                idx_src = node_to_idx[src]
                idx_tgt = node_to_idx[tgt]
                W[idx_src, idx_tgt] = link['value']
                
        # 3. Group node indices by cluster ID
        unique_clusters = list(set(node_to_cluster.values()))
        K = len(unique_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        
        cluster_node_indices = {c: [] for c in unique_clusters}
        for node_id, cluster_id in node_to_cluster.items():
            cluster_node_indices[cluster_id].append(node_to_idx[node_id])
            
        # =====================================================================
        # S_intra: Intra-Cluster Synergy 
        # Ratio of positive edge weights to total absolute edge weights within C
        # =====================================================================
        s_intra_scores = {}
        for c, indices in cluster_node_indices.items():
            if len(indices) <= 1:
                # A single node has perfect synergy by definition
                s_intra_scores[c] = 1.0 
                continue
                
            # Extract the internal routing submatrix for this cluster
            idx_mesh = np.ix_(indices, indices)
            W_c = W[idx_mesh].copy()
            
            # Zero out the diagonal (we only care about i != j)
            np.fill_diagonal(W_c, 0.0)
            
            positive_sum = np.sum(np.maximum(0, W_c))
            absolute_sum = np.sum(np.abs(W_c))
            
            if absolute_sum < epsilon:
                # If the cluster is entirely sparse internally, it doesn't violate synergy
                s_intra_scores[c] = 1.0 
            else:
                s_intra_scores[c] = positive_sum / (absolute_sum + epsilon)
                
        global_s_intra = np.mean(list(s_intra_scores.values()))
        
        # =====================================================================
        # S_inter: Inter-Cluster Homogeneity
        # Average pairwise cosine similarity of macroscopic wiring signatures
        # =====================================================================
        
        # First, build V matrix: V[i, k] is the sum of weights from node i to all nodes in cluster k
        V = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for c, indices in cluster_node_indices.items():
                k_idx = cluster_to_idx[c]
                V[i, k_idx] = np.sum(W[i, indices])
                
        s_inter_scores = {}
        for c, indices in cluster_node_indices.items():
            if len(indices) <= 1:
                # A single node is perfectly homogeneous with itself
                s_inter_scores[c] = 1.0 
                continue
                
            # Extract macroscopic wiring signatures for nodes in this cluster
            V_c = V[indices, :] # Shape: (|C|, K)
            
            # Compute pairwise cosine similarity
            # (sklearn handles zero-vectors gracefully by returning 0 similarity)
            sim_matrix = cosine_similarity(V_c)
            
            # We want the average of the upper triangle, excluding the diagonal (i != j)
            n_nodes = len(indices)
            upper_tri_indices = np.triu_indices(n_nodes, k=1)
            pairwise_sims = sim_matrix[upper_tri_indices]
            
            if len(pairwise_sims) == 0:
                s_inter_scores[c] = 1.0
            else:
                s_inter_scores[c] = np.mean(pairwise_sims)
                
        global_s_inter = np.mean(list(s_inter_scores.values()))
        
        # =====================================================================
        # Report Results
        # =====================================================================
        print(f"\n--- Clustering Evaluation ---")
        print(f"Total Clusters Evaluated: {K}")
        print(f"Global S_intra (Synergy):     {global_s_intra:.4f}  [1.0 = Perfect, no internal conflict]")
        print(f"Global S_inter (Homogeneity): {global_s_inter:.4f}  [1.0 = Perfect, identical external wiring]")
        print(f"-----------------------------\n")
        
        return {
            "global_s_intra": global_s_intra,
            "global_s_inter": global_s_inter,
            "cluster_s_intra": s_intra_scores,
            "cluster_s_inter": s_inter_scores
        }
    
    def get_layer_fallback(self, hook_name: str) -> int:
        match = re.search(r'\.(\d+)\.', hook_name)
        return int(match.group(1)) if match else 0

    def get_component_order(self, hook_name: str, layer: int) -> int:
        try:
            if layer in self.graph.connection:
                layer_map = self.graph.connection[layer]
                keys = list(layer_map.keys())
                for i, (k, _) in enumerate(keys):
                    if k.name == hook_name: return i
            if "resid" in hook_name: return 0
            if "attn" in hook_name: return 1
            if "mlp" in hook_name: return 2
            return 0
        except: return 0

    def get_logits_html(self, hook_name, feat_idx, important_output_tokens: List[Tuple[int, int]] | None = None):
        try:
            sae = self.sae_dict.get(hook_name)
            if not sae: return "N/A"
            if hasattr(sae, 'W_dec'): vec = sae.W_dec[feat_idx]
            elif hasattr(sae, 'decoder'): vec = sae.decoder.weight.data.T[feat_idx]
            else: return "N/A"
            logits = self.model.ln_final(vec) @ self.model.W_U
            k = 5
            pos_vals, pos_ids = logits.topk(k)
            neg_vals, neg_ids = logits.topk(k, largest=False)
            
            def pill(t, v, top, s=None):
                c = "background:#d4edda;color:#155724" if top else "background:#f8d7da;color:#721c24"
                s_str = f" Seq: {s}" if s is not None else ""
                return f'<span style="display:inline-block;padding:2px 6px;margin:2px;border-radius:4px;font-size:10px;{c}" title="{v:.4f}{s_str}">{html.escape(t)} : {v:.2f}</span>'
            
            h = "<div><div style='font-size:10px;color:#666'>Top</div>"
            for t_str, v in zip(self.model.to_str_tokens(pos_ids), pos_vals): h += pill(t_str, v, True)
            h += "<div style='font-size:10px;color:#666;margin-top:4px'>Bottom</div>"
            for t_str, v in zip(self.model.to_str_tokens(neg_ids), neg_vals): h += pill(t_str, v, False)
            if important_output_tokens is not None:
                h += "<div style='font-size:10px;color:#666;margin-top:4px'>Output Tokens</div>"
                important_tokens_vals = []
                important_tokens_ids = []
                important_tokens_seqs = []
                for (s, idx) in important_output_tokens:
                    important_tokens_vals.append(logits[idx].item())
                    important_tokens_ids.append(idx)
                    important_tokens_seqs.append(s)
                for t_str, v, s in zip(self.model.to_str_tokens(t.tensor(important_tokens_ids)), important_tokens_vals, important_tokens_seqs): 
                    h += pill(t_str, v, v > 0, s)
            h += "</div>"
            return h
        except: return "Error"

    def build_graph_json(self, override: bool = False, skip_edges: bool = False, n_intervals: int = 5, k_per_interval: int = 5, buffer: int = 10, compute_edge=True, compute_html: bool = True):
        # compute_html=False skips the per-node act/logits HTML strings (only used by the interactive
        # dashboards and the `llm` clustering method). The activation fetch and `density` filter are
        # kept, so the node set, links and every metric field are identical -- headless evals set it
        # False to avoid building HTML they never read.
        if self.graph_json is not None and not override:
            return self.graph_json
        
        nodes_list, raw_links_list = [], []
        def get_id(hook, seq, feat, is_err): return f"{hook}::{seq}::{'err' if is_err else 'feat'}::{feat}"
        
        self.get_feature_acts()

        for (node_key, _), sparse_act in self.graph.nodes.items():
            hook = node_key.name
            layer = -1
            for cur_layer, dict_connection in self.graph.connection.items():
                for (k, _) in dict_connection.keys():
                    if k.name == hook:
                        layer = cur_layer
                        break
                if layer != -1: break
            if layer == -1: layer = self.get_layer_fallback(hook)

            comp = self.get_component_order(hook, layer)
            scores = self.graph.node_scores.get((node_key, _))
            
            if sparse_act.act is not None:
                for s, f in zip(*sparse_act.act.nonzero(as_tuple=True)):
                    s_item, f_item = s.item(), f.item()
                    node_id = get_id(hook, s_item, f_item, False)
                    cluster_id = self.node_clusters.get(node_id, None)
                    
                    density = 0
                    node_html = ""
                    # Optional: compute html/density if needed for detailed view
                    # We do this here so aggregation has access to it
                    if self.data_toks is not None and self.full_data_dict and hook in self.full_data_dict:
                        values, indices = self.full_data_dict[hook]
                        try:
                            data = fetch_feature_activation_intervals(self.model, f_item, values, indices, self.data_toks, n_intervals, k_per_interval, buffer)
                            node_html = generate_interactive_html_embeddable(data) if compute_html else ""
                            density = data['density'] if 'density' in data else 0
                        except:
                            node_html = "<div style='color:#999;font-style:italic'>No activation data.</div>"
                    else:
                        node_html = "<div style='color:#999;font-style:italic'>No activation data.</div>"

                    # (Optional filter if you want to keep it)
                    if density > 0.1: continue 
                    
                    if not compute_html:
                        logit_html = ""
                    elif self.important_output_tokens is not None:
                        logit_html = self.get_logits_html(
                            hook, f_item, [(tok_s, tok_idx) for tok_s, tok_idx in self.important_output_tokens if tok_s >= s_item]
                        )
                    else:
                        logit_html = self.get_logits_html(hook, f_item)

                    nodes_list.append({
                        "id": node_id, "label": f"{f_item}",
                        "type": "feature", "layer": layer, "comp": comp, "seq": s_item, "seq_str": self.tokens[s_item],
                        "score": scores.act[s_item, f_item].item() if scores else 0.0,
                        "hook": hook, "feat_idx": f_item,
                        "cluster": cluster_id,
                        "logits_html": logit_html,
                        "act_html": node_html,
                        "density": density,
                    })
            if sparse_act.resc is not None:
                for s in sparse_act.resc.nonzero(as_tuple=True)[0]:
                    s_item = s.item()
                    nodes_list.append({
                        "id": get_id(hook, s_item, 0, True), "label": "Err",
                        "type": "error", "layer": layer, "comp": comp, "seq": s_item, "seq_str": self.tokens[s_item],
                        "score": scores.resc[s_item].item() if scores else 0.0,
                        "hook": hook, "feat_idx": -1, "logits_html": "", "act_html": "", "density": -1,
                    })
        
        if not skip_edges:
            for (end_key, end_idx), start_dict in self.graph.edges.items():
                for (start_key, start_idx), edge_tensor in start_dict.items():
                    indices = edge_tensor.coalesce().indices()
                    mask_values = edge_tensor.coalesce().values()
                    values = self.graph.edge_scores[(end_key, end_idx)][(start_key, start_idx)].coalesce().values()
                    d_start = values.shape[2] - 1
                    
                    # --- OPTIMIZATION 1: Lift node lookup out of the inner loop ---
                    src_node = self.graph.nodes[(start_key, start_idx)]
                    start_name = start_key.name
                    end_name = end_key.name
                    
                    # --- OPTIMIZATION 2: Pre-extract active features and errors ONCE ---
                    active_feats = []
                    if src_node.act is not None:
                        f_seq, f_idx = src_node.act.nonzero(as_tuple=True)
                        # Convert to fast Python lists/tuples for the inner loop
                        active_feats = tuple(zip(f_seq.tolist(), f_idx.tolist()))
                        
                    active_errs = []
                    if src_node.resc is not None:
                        e_seq = src_node.resc.nonzero(as_tuple=True)[0]
                        active_errs = e_seq.tolist()
                    # -----------------------------------------------------------------

                    for i in range(indices.shape[1]):
                        end_seq, end_feat = indices[0, i].item(), indices[1, i].item()
                        tgt_id = get_id(end_name, end_seq, end_feat, False)
                        
                        # 1. Loop only through confirmed active features
                        for src_seq, src_feat in active_feats:
                            weight = values[i, src_seq, src_feat].item()
                            mask = mask_values[i, src_seq, src_feat].item()
                            if weight != 0.0 and mask:  # Only append if an edge actually exists
                                src_id = get_id(start_name, src_seq, src_feat, False)
                                raw_links_list.append({"source": src_id, "target": tgt_id, "value": weight})
                                
                        # 2. Loop only through confirmed active errors
                        for src_seq in active_errs:
                            weight = values[i, src_seq, d_start].item()
                            mask = mask_values[i, src_seq, d_start].item()
                            if weight != 0.0 and mask:
                                src_id = get_id(start_name, src_seq, 0, True)
                                raw_links_list.append({"source": src_id, "target": tgt_id, "value": weight})

        valid = set(n['id'] for n in nodes_list)
        clean_links = [l for l in raw_links_list if l['source'] in valid and l['target'] in valid]
        
        max_comps = max((n['comp'] for n in nodes_list), default=0) + 1
        max_node_score = max((abs(n['score']) for n in nodes_list), default=1.0)

        self.graph_json = {
            "nodes": nodes_list, "links": clean_links, "prompt": self.prompt_str,
            "layers": sorted(list(set(n["layer"] for n in nodes_list))),
            "tokens": self.tokens,
            "max_comps": max(3, max_comps),
            "max_node_score": max_node_score
        }
        return self.graph_json
    
    def build_aggregate_graph_json(self, override: bool = False, n_intervals: int = 5, k_per_interval: int = 5, buffer: int = 10):
        """
        Generates a Super-Node graph where nodes are clusters and edges are averaged weights.
        """
        if self.compressed_graph_json is not None and not override:
            return self.compressed_graph_json
        
        raw_data = self.build_graph_json(n_intervals=n_intervals, k_per_interval=k_per_interval, buffer=buffer)
        
        # 1. Aggregate Nodes
        # Map: cluster_id -> { stats... }
        clusters = {}
        # New: Create map for easy access to populate outgoing links later
        node_map = {n['id']: n for n in raw_data['nodes']}
        for n in raw_data['nodes']:
            n['outgoing'] = [] # Initialize outgoing for all nodes
            n['incoming'] = [] # Initialize incoming for all nodes
            
            cid = n.get('cluster')
            if cid is not None:
                if cid not in clusters:
                    clusters[cid] = {
                        'id': f"cluster_{cid}",
                        'cluster_label': cid,
                        'features': [],
                        'layer_sum': 0,
                        'score_sum': 0,
                        'seq_sum': 0,
                        'count': 0,
                        'type': 'super_node'
                    }
                c = clusters[cid]
                c['features'].append(n)
                c['layer_sum'] += n['layer']
                c['score_sum'] += n['score']
                c['seq_sum'] += n['seq']
                c['count'] += 1
            else:
                pass
        
        super_nodes = []
        node_id_to_cluster_id = {}
        
        for cid, c in clusters.items():
            avg_layer = c['layer_sum'] / c['count']
            # avg_score = c['score_sum'] / c['count']
            avg_score = c['score_sum']
            avg_seq = c['seq_sum'] / c['count']
            
            c['layer'] = avg_layer
            c['score'] = avg_score
            c['seq'] = avg_seq
            
            # Sort features by score desc
            c['features'].sort(key=lambda x: abs(x['score']), reverse=True)
            super_nodes.append(c)
            for f in c['features']:
                node_id_to_cluster_id[f['id']] = c['id']

        # 2. Aggregate Edges and Populate Outgoing
        # Map: (source_cid, target_cid) -> {weight_sum: 0, count: 0}
        edge_map = defaultdict(lambda: {'weight_sum': 0.0, 'count': 0})
        
        for l in raw_data['links']:
            sid = l['source']
            tid = l['target']
            
            scid = node_id_to_cluster_id.get(sid)
            tcid = node_id_to_cluster_id.get(tid)
            
            if scid and tcid and scid != tcid:
                key = (scid, tcid)
                edge_map[key]['weight_sum'] += l['value']
                edge_map[key]['count'] += 1
            
            # Populate fine-grained outgoing
            if sid in node_map and tid in node_map:
                s_node = node_map[sid]
                t_node = node_map[tid]
                t_cluster_raw = t_node.get('cluster') # Get the integer cluster ID
                s_cluster_raw = s_node.get('cluster') # Get the integer cluster ID for incoming
                
                s_node['outgoing'].append({
                    'id': tid,
                    'feat_idx': t_node['feat_idx'],
                    'score': l['value'],
                    'cluster': t_cluster_raw, 
                    'hook': t_node['hook']
                })
                
                t_node['incoming'].append({
                    'id': sid,
                    'feat_idx': s_node['feat_idx'],
                    'score': l['value'],
                    'cluster': s_cluster_raw,
                    'hook': s_node['hook']
                })
                
        super_links = []
        for (scid, tcid), stats in edge_map.items():
            # Average weight of ACTIVE connections
            # avg_weight = stats['weight_sum'] / stats['count']
            avg_weight = stats['weight_sum']
            
            super_links.append({
                'source': scid,
                'target': tcid,
                'value': avg_weight,
                'count': stats['count'] # Keep track of how many edges formed this link
            })

        # 3. Final JSON payload
        max_node_score = max((abs(n['score']) for n in super_nodes), default=1.0)
        
        # We need layers for Y-axis (min/max)
        if super_nodes:
            min_l = min(n['layer'] for n in super_nodes)
            max_l = max(n['layer'] for n in super_nodes)
        else:
            min_l, max_l = 0, 0
            
        self.compressed_graph_json = {
            "nodes": super_nodes,
            "links": super_links,
            "prompt": self.prompt_str,
            "layers": [min_l, max_l], 
            "max_node_score": max_node_score,
            "tokens": self.tokens 
        }
        return self.compressed_graph_json

    def generate_html(self, output_path="sae_graph.html", override=False):
        data = self.build_graph_json(override)
        class NaNEncoder(json.JSONEncoder):
            def default(self, o): return 0 if isinstance(o, float) and math.isnan(o) else super().default(o)
        json_data = json.dumps(data, cls=NaNEncoder)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>SAE Circuit</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {{ margin:0; font-family: sans-serif; display: flex; height: 100vh; overflow: hidden; }}
        
        /* New Left Sidebar for Clusters */
        #cluster-nav {{ width: 50px; background: #f8f9fa; border-right: 1px solid #ddd; display: none; flex-direction: column; align-items: center; padding: 10px 0; overflow-y: auto; z-index: 20; }}
        .cluster-item {{ width: 30px; height: 30px; border-radius: 50%; margin-bottom: 8px; cursor: pointer; display: flex; justify-content: center; align-items: center; font-size: 10px; font-weight: bold; color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.2); transition: transform 0.1s; border: 2px solid transparent; flex-shrink: 0; }}
        .cluster-item:hover {{ transform: scale(1.1); }}
        .cluster-item.active {{ border-color: #333; transform: scale(1.1); box-shadow: 0 0 0 2px #fff, 0 0 0 4px #333; }}

        #sidebar {{ width: 550px; border-left: 1px solid #ddd; background: #fff; overflow-y: auto; z-index:10; box-shadow: -2px 0 5px rgba(0,0,0,0.05); }}
        .panel {{ padding: 15px; border-bottom: 1px solid #eee; }}
        .panel h3 {{ margin: 0 0 10px 0; font-size: 14px; text-transform: uppercase; color: #555; }}
        
        .panel-row {{ display: flex; border-bottom: 1px solid #eee; }}
        .panel-col {{ flex: 1; padding: 10px; min-width: 0; }}
        .panel-col:first-child {{ border-right: 1px solid #eee; }}
        .panel-col h3 {{ margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase; color: #555; font-weight:bold; }}

        #viz {{ flex-grow: 1; position: relative; background: #fdfdfd; }}
        .prompt-box {{ position: absolute; top:10px; left:10px; background: rgba(255,255,255,0.9); padding:8px; border:1px solid #ccc; font-family: monospace; font-size: 11px; max-width: 600px; z-index: 5; }}
        
        .node circle {{ fill: #fff; stroke: #888; stroke-width: 1.5px; transition: all 0.2s; }}
        .node.feature circle {{ stroke: #28a745; fill: #eafbe7; }}
        .node.error rect {{ stroke: #fd7e14; fill: #fff3e0; }}
        .node:hover circle, .node:hover rect {{ stroke: #000; stroke-width: 2px; }}
        .node.selected circle, .node.selected rect {{ stroke: #d00; stroke-width: 3px; }}
        
        .link {{ fill: none; stroke: #999; stroke-opacity: 0.2; }}
        .link.highlight {{ stroke: #333; stroke-opacity: 0.8; stroke-width: 1.5px; }}
        
        /* Adjusted feat-item to prevent overflow in narrow columns */
        .feat-item {{ display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid #f8f8f8; cursor: pointer; font-size: 11px; }}
        .feat-item:hover {{ background: #f0f0f0; }}
        .f-info {{ display: flex; flex-direction: column; overflow: hidden; flex: 1 1 auto; margin-right: 5px; min-width: 0; }}
        .f-main {{ font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: flex; align-items: center; }}
        .f-meta {{ font-size: 9px; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .f-bar-con {{ display: flex; align-items: center; width: 50px; justify-content: flex-end; flex: 0 0 auto; }}
        .f-bar-bg {{ width: 30px; height: 6px; background: #eee; margin-left: 3px; }}
        .f-bar {{ height: 100%; }}
        
        .axis text {{ font-family: monospace; font-size: 10px; fill: #666; }}
        .axis path, .axis line {{ stroke: #ddd; }}
        
        textarea:focus {{ outline: 1px solid #28a745; }}
    </style>
</head>
<body>
<div id="cluster-nav"></div>
<div id="viz">
    <div class="prompt-box"><strong>Prompt:</strong> {self.prompt_str}</div>
</div>
<div id="sidebar">
    <div class="panel">
        <h2 id="sb-title" style="margin:0; font-size:18px;">Select a Node</h2>
        <div id="sb-meta" style="font-size:12px; color:#666; margin-top:5px;"></div>
    </div>
    <div id="sb-content" style="display:none">
        <div class="panel-row">
            <div class="panel-col"><h3>Inputs</h3><div id="list-in"></div></div>
            <div class="panel-col"><h3>Outputs</h3><div id="list-out"></div></div>
        </div>
        <div class="panel"><h3>Logits</h3><div id="logits"></div></div>
        <div class="panel"><h3>Activations</h3><div id="activations"></div></div>
    </div>
</div>
<script>
    const data = {json_data};
    const width = document.getElementById('viz').clientWidth;
    const height = document.getElementById('viz').clientHeight;
    
    // Default D3 categorical colors
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // SETUP CLUSTER NAV (LEFT SIDEBAR)
    // Extract unique clusters
    const clusters = [...new Set(data.nodes.map(d => d.cluster).filter(c => c !== null))].sort((a,b)=>a-b);
    let activeCluster = null;

    if (clusters.length > 0) {{
        const nav = document.getElementById("cluster-nav");
        nav.style.display = "flex";
        
        clusters.forEach(c => {{
            const item = document.createElement("div");
            item.className = "cluster-item";
            item.style.backgroundColor = colorScale(c);
            item.innerText = c;
            item.title = "Cluster " + c;
            item.onclick = (e) => toggleCluster(c, item);
            nav.appendChild(item);
        }});
    }}

    function toggleCluster(c, element) {{
        // Reset Node Selection View first
        d3.selectAll(".node").classed("selected", false);
        d3.selectAll(".link").classed("highlight", false).style("stroke-opacity", 0.2);
        document.getElementById("sb-content").style.display = "none";
        document.getElementById("sb-title").innerText = "Select a Node";
        document.getElementById("sb-meta").innerText = "";

        if (activeCluster === c) {{
            // Deselect logic
            activeCluster = null;
            d3.selectAll(".cluster-item").classed("active", false);
            // Reset opacity to default
            d3.selectAll(".node")
                .style("opacity", 1)
                .style("stroke-width", null)
                .style("stroke", null);
        }} else {{
            // Select logic
            activeCluster = c;
            d3.selectAll(".cluster-item").classed("active", false);
            element.classList.add("active");
            
            // Brighten selected cluster, dim others
            d3.selectAll(".node").each(function(d) {{
                const isMember = d.cluster === c;
                d3.select(this)
                    .style("opacity", isMember ? 1 : 0.1)
                    .style("stroke-width", isMember ? "2.5px" : null)
                    .style("stroke", isMember ? colorScale(c) : null);
            }});
        }}
    }}

    if(!data.nodes.length) {{ document.getElementById('viz').innerHTML = "No nodes found."; }}
    else {{
        const svg = d3.select("#viz").append("svg").attr("width","100%").attr("height","100%")
            .call(d3.zoom().on("zoom", e => g.attr("transform", e.transform))).append("g");
        const g = svg;
        
        const margin = {{top: 50, right: 100, bottom: 80, left: 60}};
        const layers = data.layers.length ? data.layers : [0];
        const maxNodeScore = data.max_node_score || 1.0;
        
        const xScale = d3.scaleLinear()
            .domain([0, data.tokens.length - 1]) 
            .range([margin.left, width - margin.right]);
            
        const minL = Math.min(...layers), maxL = Math.max(...layers);
        const yScale = d3.scaleLinear()
            .domain([minL - 0.5, maxL + 1.5]) 
            .range([height - margin.bottom, margin.top]);

        const xAxis = d3.axisBottom(xScale)
            .tickValues(d3.range(0, data.tokens.length)) 
            .tickFormat(i => data.tokens[i] || i);
            
        g.append("g").attr("class", "axis")
            .attr("transform", `translate(0, ${{height - margin.bottom}})`)
            .call(xAxis)
            .selectAll("text")
            .style("text-anchor", "start") 
            .attr("dx", "10px") 
            .attr("dy", "-2px")
            .attr("transform", "rotate(45)");

        g.append("text").attr("x", width/2).attr("y", height-10).attr("text-anchor", "middle")
         .attr("fill", "#999").attr("font-size","10px").text("Sequence Position");

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).strength(0.01)) 
            .force("charge", d3.forceManyBody().strength(-100))
            .force("collide", d3.forceCollide(8).strength(1))
            .force("x", d3.forceX(d => {{
                return xScale(d.seq);
            }}).strength(2.0))
            .force("y", d3.forceY(d => {{
                const compOffset = (d.comp || 0) / (data.max_comps || 3) * 0.8;
                return yScale(d.layer + compOffset);
            }}).strength(2.0));

        const link = g.append("g").selectAll("path").data(data.links).join("path")
            .attr("class", "link")
            .attr("d", d => {{
                // Fallback for string vs object IDs if D3 hasn't processed them yet
                const srcX = d.source.x || 0, srcY = d.source.y || 0;
                const tgtX = d.target.x || 0, tgtY = d.target.y || 0;
                const dx = tgtX - srcX;
                return `M${{srcX}},${{srcY}} C${{srcX + dx/2}},${{srcY}} ${{tgtX - dx/2}},${{tgtY}} ${{tgtX}},${{tgtY}}`;
            }});

        const node = g.append("g").selectAll("g").data(data.nodes).join("g")
            .attr("class", d => "node " + d.type)
            .call(d3.drag()
                .on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }})
            )
            .on("click", (e,d) => selectNode(d));

        node.append("title")
            .text(d => `Feature: ${{d.type==='error'?'Error':d.feat_idx}}\\nHook: ${{d.hook}}\\nSeq: ${{d.seq}}\\nSeq_str: ${{d.seq_str}}\\nScore: ${{d.score.toFixed(5)}}`);

        node.filter(d=>d.type==='feature').append("circle")
            .attr("r", 5)
            .style("stroke", d => d.cluster !== null ? colorScale(d.cluster) : "#28a745")
            .style("fill", d => d.cluster !== null ? d3.color(colorScale(d.cluster)).copy({{opacity:0.2}}) : "#eafbe7");
            
        node.filter(d=>d.type==='error').append("rect").attr("x",-4).attr("y",-4).attr("width",8).attr("height",8).attr("transform","rotate(45)");

        simulation.on("tick", () => {{
            link.attr("d", d => {{
                if (isNaN(d.source.x) || isNaN(d.target.x)) return "";
                const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
                return `M${{d.source.x}},${{d.source.y}} C${{d.source.x + dx/2}},${{d.source.y}} ${{d.target.x - dx*0.5}},${{d.target.y}} ${{d.target.x}},${{d.target.y}}`;
            }});
            node.attr("transform", d => {{
                if (isNaN(d.x) || isNaN(d.y)) return "translate(0,0)";
                return `translate(${{d.x}},${{d.y}})`
            }});
        }});

        function selectNode(d) {{
            // Clear any active cluster selection when selecting a specific node
            activeCluster = null;
            d3.selectAll(".cluster-item").classed("active", false);

            // 1. Reset all Highlights
            d3.selectAll(".node").classed("selected", false).style("opacity", 0.4).style("stroke-width", null).style("stroke", null);
            
            // Re-apply base colors for clusters if needed (clean slate)
            d3.selectAll(".node").filter(n => n.cluster !== null).style("stroke", n => colorScale(n.cluster));

            d3.selectAll(".link").classed("highlight", false).style("stroke-opacity", 0.2);
            
            // 2. Highlight Current Node
            const sel = d3.select(event.currentTarget);
            sel.classed("selected", true).style("opacity", 1);
            
            // 3. Highlight Cluster Peers (if any) - "Brighten them up"
            if (d.cluster !== null) {{
                d3.selectAll(".node").filter(n => n.cluster === d.cluster)
                    .style("opacity", 1)
                    .style("stroke-width", "2.5px")
                    .style("stroke", colorScale(d.cluster)); // Reinforce color
            }}
            
            // 4. Highlight Connections
            const connectedLink = link.filter(l => (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id);
            connectedLink.classed("highlight", true).style("stroke-opacity", 0.8);
            
            // Also brighten connected nodes
            connectedLink.each(l => {{
               const sid = l.source.id || l.source;
               const tid = l.target.id || l.target;
               d3.selectAll(".node").filter(n => n.id === sid || n.id === tid).style("opacity", 1);
            }});

            // 5. Sidebar Updates
            const scW = Math.min(Math.abs(d.score) / maxNodeScore * 50, 50);
            const scoreBarHtml = `<div style="display:inline-block; width:50px; height:6px; background:#eee; margin-left:5px; vertical-align:middle;"><div style="width:${{scW}}px; height:100%; background:#555"></div></div>`;
            
            // Add a "Bar Tab" (colored strip) to the header if cluster exists
            let clusterHtml = "";
            if (d.cluster !== null) {{
                const cColor = colorScale(d.cluster);
                clusterHtml = `<br><span style="border-left: 4px solid ${{cColor}}; padding-left: 6px;">Cluster: <span style='color:${{cColor}}; font-weight:bold;'>${{d.cluster}}</span></span>`;
            }}
            
            document.getElementById("sb-title").innerHTML = d.type==='error' ? "Error Node" : `Feature ${{d.feat_idx}}`;
            document.getElementById("sb-meta").innerHTML = `Hook: ${{d.hook}}<br>Seq: ${{d.seq}}<br>Seq_str: ${{d.seq_str}}<br>Score: ${{d.score.toFixed(5)}} ${{scoreBarHtml}} ${{clusterHtml}}`;
            document.getElementById("sb-content").style.display = "block";
            document.getElementById("logits").innerHTML = d.logits_html;
            document.getElementById("activations").innerHTML = d.act_html;
            
            // Safe ID check for filtering
            updateList("list-in", l => (l.target.id || l.target) === d.id, l => l.source);
            updateList("list-out", l => (l.source.id || l.source) === d.id, l => l.target);
        }}

        function updateList(id, filter, getOther) {{
            const div = document.getElementById(id);
            div.innerHTML = "";
            const items = data.links.filter(filter);
            const sumEdgeVal = items.reduce((sum, item) => sum + Math.abs(item.value), 0) || 1e-9;
            items.sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
            
            if(!items.length) div.innerHTML = "<div style='color:#999;font-size:11px'>None</div>";
            
            items.forEach(l => {{
                // Handle both object reference and string ID
                let other = getOther(l);
                const otherId = other.id || other;
                const otherNode = data.nodes.find(n => n.id === otherId);
                
                if(!otherNode) return;
                const row = document.createElement("div");
                row.className = "feat-item";
                
                const pct = (Math.abs(l.value) / sumEdgeVal) * 100;
                const w = Math.min(pct, 100); 
                const col = l.value > 0 ? "#28a745" : "#dc3545";
                const nScW = Math.min((Math.abs(otherNode.score) / maxNodeScore) * 20, 20);
                
                // Add Small Cluster Tab/Dot if neighbor has a cluster
                let clusterDot = "";
                if(otherNode.cluster !== null) {{
                    clusterDot = `<span title="Cluster ${{otherNode.cluster}}" style="display:inline-block; width:6px; height:6px; border-radius:50%; background:${{colorScale(otherNode.cluster)}}; margin-right:4px;"></span>`;
                }}
                
                row.innerHTML = `
                    <div class="f-info">
                        <span class="f-main" title="${{otherNode.hook}}">
                            ${{clusterDot}}
                            ${{otherNode.type==='error'?'Err':otherNode.feat_idx}} 
                            <div style="display:inline-block; width:20px; height:4px; background:#eee; vertical-align:middle; margin-left:4px;"><div style="width:${{nScW}}px; height:100%; background:#888;"></div></div>
                        </span>
                        <span class="f-meta">${{otherNode.hook.split('.').slice(-2).join('.')}} • Seq ${{otherNode.seq}}</span>
                    </div>
                    <div class="f-bar-con">
                        <span style="font-family:monospace;color:${{col}}">${{l.value.toFixed(4)}}</span>
                        <div class="f-bar-bg"><div class="f-bar" style="width:${{w}}%; background:${{col}}"></div></div>
                    </div>
                `;
                row.onclick = () => {{
                    const n = d3.selectAll(".node").filter(dn => dn.id === otherNode.id);
                    if(!n.empty()) {{ n.dispatch("click"); }}
                }};
                div.appendChild(row);
            }});
        }}
    }}
</script></body></html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"Visualization saved to {output_path}")
        
    def generate_aggregated_html(self, output_path="sae_graph_aggregated.html", override=True):
        self.build_aggregate_graph_json(override)
        
        class NaNEncoder(json.JSONEncoder):
            def default(self, o): return 0 if isinstance(o, float) and math.isnan(o) else super().default(o)
            
        json_data = json.dumps(self.compressed_graph_json, cls=NaNEncoder)

        # 4. HTML Template
        # Similar D3 but sidebar logic allows drilling down
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Super-Node Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {{ margin:0; font-family: sans-serif; display: flex; height: 100vh; overflow: hidden; }}
        #sidebar {{ width: 500px; border-left: 1px solid #ddd; background: #fff; overflow-y: auto; z-index:10; box-shadow: -2px 0 5px rgba(0,0,0,0.05); display: flex; flex-direction: column; }}
        
        .header-panel {{ padding: 15px; border-bottom: 1px solid #eee; background: #fafafa; }}
        .header-panel h2 {{ margin: 0; font-size: 18px; color: #333; }}
        .header-meta {{ font-size: 12px; color: #666; margin-top: 5px; }}
        
        .content-panel {{ flex: 1; overflow-y: auto; padding: 0; }}
        .section-title {{ background: #eee; padding: 5px 10px; font-size: 11px; font-weight: bold; color: #555; text-transform: uppercase; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; }}
        
        .list-item {{ padding: 8px 10px; border-bottom: 1px solid #f0f0f0; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
        .list-item:hover {{ background: #f0f8ff; }}
        
        .feat-row {{ display: flex; flex-direction: column; width: 100%; }}
        .feat-head {{ font-weight: 500; font-size: 13px; }}
        .feat-sub {{ font-size: 10px; color: #888; margin-top: 2px; }}
        
        .edge-val {{ font-family: monospace; font-size: 11px; }}
        .pos {{ color: #28a745; }} .neg {{ color: #dc3545; }}

        #viz {{ flex-grow: 1; position: relative; background: #fdfdfd; }}
        
        /* Drill Down View */
        #feature-view {{ display: none; padding: 15px; background: #fff; height: 100%; }}
        .back-btn {{ margin-bottom: 10px; padding: 5px 10px; background: #eee; border: none; cursor: pointer; font-size: 12px; border-radius: 4px; }}
        .back-btn:hover {{ background: #ddd; }}

        .node circle {{ stroke: #fff; stroke-width: 2px; transition: all 0.2s; }}
        .node:hover circle {{ stroke: #333; }}
        .node.selected circle {{ stroke: #000; stroke-width: 3px; }}
        .node text {{ font-size: 10px; pointer-events: none; text-shadow: 0 1px 0 #fff; }}

        .link {{ stroke: #999; stroke-opacity: 0.3; fill: none; }}
        .link.highlight {{ stroke: #333; stroke-opacity: 0.8; }}
        
        .axis text {{ font-family: monospace; font-size: 10px; fill: #666; }}
        .axis path, .axis line {{ stroke: #ddd; }}
        
        textarea {{ width:100%; height:60px; font-family:sans-serif; font-size:12px; border:1px solid #ddd; padding:4px; resize:vertical; box-sizing:border-box; margin-top:5px; }}
        textarea:focus {{ outline: 1px solid #28a745; }}
    </style>
</head>
<body>
<div id="viz"></div>
<div id="sidebar">
    <div id="cluster-view">
        <div class="header-panel">
            <h2 id="sb-title">Select a Cluster</h2>
            <div id="sb-meta" class="header-meta"></div>
            <!-- Added Note Area for Cluster -->
            <div style="margin-top:10px;">
                <label style="font-size:10px; font-weight:bold; color:#888;">CAPTION / NOTES</label>
                <textarea id="cluster-notes" placeholder="Write caption for this cluster..."></textarea>
            </div>
        </div>
        <div class="content-panel">
            <div class="section-title">Cluster Features</div>
            <div id="list-features"></div>
            
            <div class="section-title">Inputs (Source Clusters)</div>
            <div id="list-in"></div>
            
            <div class="section-title">Outputs (Target Clusters)</div>
            <div id="list-out"></div>
        </div>
    </div>
    <div id="feature-view">
        <button class="back-btn" onclick="showClusterView()">← Back to Cluster</button>
        <h3 id="feat-title">Feature Info</h3>
        <!-- Added Note Area for Feature -->
        <div style="margin-bottom:10px;">
             <textarea id="feat-notes" placeholder="Write caption for this feature..."></textarea>
        </div>
        <div id="feat-content"></div>
    </div>
</div>

<script>
    const data = {json_data};
    const width = document.getElementById('viz').clientWidth;
    const height = document.getElementById('viz').clientHeight;
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    if(!data.nodes.length) {{ document.getElementById('viz').innerHTML = "No clustered nodes found."; }}
    else {{
        const svg = d3.select("#viz").append("svg").attr("width","100%").attr("height","100%")
            .call(d3.zoom().on("zoom", e => g.attr("transform", e.transform))).append("g");
        const g = svg;
        
        // Y-Scale based on Layer
        const margin = {{top: 50, right: 50, bottom: 80, left: 60}};
        const [minL, maxL] = data.layers;
        const yScale = d3.scaleLinear().domain([minL, maxL]).range([height-margin.bottom, margin.top]);
        
        // X-Scale based on Sequence Position (tokens)
        const xScale = d3.scaleLinear()
            .domain([0, data.tokens.length - 1]) 
            .range([margin.left, width - margin.right]);
            
        // Axis
        const xAxis = d3.axisBottom(xScale)
            .tickValues(d3.range(0, data.tokens.length)) 
            .tickFormat(i => data.tokens[i] || i);
            
        g.append("g").attr("class", "axis")
            .attr("transform", `translate(0, ${{height - margin.bottom}})`)
            .call(xAxis)
            .selectAll("text")
            .style("text-anchor", "start") 
            .attr("dx", "10px") 
            .attr("dy", "-2px")
            .attr("transform", "rotate(45)");
            
        g.append("text").attr("x", width/2).attr("y", height-10).attr("text-anchor", "middle")
         .attr("fill", "#999").attr("font-size","10px").text("Sequence Position");
        
        // Force Layout
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).strength(0.02))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("collide", d3.forceCollide(d => Math.sqrt(d.features.length)*4 + 5).strength(1))
            .force("y", d3.forceY(d => yScale(d.layer)).strength(1.5))
            .force("x", d3.forceX(d => xScale(d.seq)).strength(2.0)); // ALIGN BY AVG SEQ

        const link = g.append("g").selectAll("path").data(data.links).join("path")
            .attr("class", "link")
            .attr("stroke-width", d => Math.min(Math.sqrt(Math.abs(d.value))*10, 10)); // Thickness by weight

        const node = g.append("g").selectAll("g").data(data.nodes).join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }})
            )
            .on("click", selectCluster);

        // Circle size based on number of features in cluster
        node.append("circle")
            .attr("r", d => Math.sqrt(d.features.length)*4 + 5)
            .attr("fill", d => colorScale(d.cluster_label));
            
        node.append("text")
            .attr("dy", ".35em")
            .attr("text-anchor", "middle")
            .attr("fill", "#fff")
            .text(d => d.cluster_label);

        // --- NEW: Annotation Layer for Captions ---
        const annotationLayer = g.append("g").attr("class", "annotations-layer");
        let annotationGroups;

        function updateAnnotations() {{
            const annotatedNodes = data.nodes.filter(d => d.notes && d.notes.trim() !== "");
            
            annotationGroups = annotationLayer.selectAll(".annotation")
                .data(annotatedNodes, d => d.id);
                
            const enter = annotationGroups.enter().append("g")
                .attr("class", "annotation")
                .style("cursor", "move")
                .call(d3.drag()
                    .on("drag", (e, d) => {{
                        d.note_dx = (d.note_dx || 0) + e.dx;
                        d.note_dy = (d.note_dy || 0) + e.dy;
                        updateAnnotationsPos();
                    }})
                );
                
            enter.append("rect")
                .attr("fill", "rgba(255, 255, 255, 0.9)")
                .attr("stroke", "#888")
                .attr("stroke-width", "1px")
                .attr("rx", 3).attr("ry", 3);
                
            enter.append("text")
                .attr("font-size", "11px")
                .attr("font-family", "sans-serif")
                .attr("fill", "#222")
                .attr("dy", "1em")
                .attr("dx", "4px")
                .style("pointer-events", "none");
                
            annotationGroups = enter.merge(annotationGroups);
            
            annotationGroups.select("text")
                .text(d => d.notes)
                .each(function(d) {{
                    d.bbox = this.getBBox();
                }});
                
            annotationGroups.select("rect")
                .attr("width", d => d.bbox ? d.bbox.width + 8 : 0)
                .attr("height", d => d.bbox ? d.bbox.height + 4 : 0)
                .attr("x", 0)
                .attr("y", 0);
                
            annotatedNodes.forEach(d => {{
                if (d.note_dx === undefined) d.note_dx = 20;
                if (d.note_dy === undefined) d.note_dy = -20;
            }});
            
            annotationGroups.exit().remove();
            updateAnnotationsPos();
        }}

        function updateAnnotationsPos() {{
            if (!annotationGroups) return;
            annotationGroups.attr("transform", d => `translate(${{d.x + (d.note_dx || 0)}},${{d.y + (d.note_dy || 0)}})`);
        }}
        
        // Initialize annotations if any are saved in the data
        updateAnnotations();

        simulation.on("tick", () => {{
            link.attr("d", d => {{
                return `M${{d.source.x}},${{d.source.y}} L${{d.target.x}},${{d.target.y}}`;
            }});
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
            updateAnnotationsPos(); // Keep annotations attached
        }});

        function selectCluster(e, d) {{
            // Highlight
            d3.selectAll(".node").classed("selected", false);
            d3.select(this).classed("selected", true);
            
            // Sidebar Info
            document.getElementById("sb-title").innerText = `Cluster ${{d.cluster_label}}`;
            document.getElementById("sb-meta").innerHTML = `
                Size: ${{d.features.length}} Features<br>
                Layer: ${{d.layer.toFixed(2)}}<br>
                Score: ${{d.score.toFixed(4)}}<br>
                Seq: ${{d.seq.toFixed(2)}}
            `;
            
            // Update Note box
            const noteBox = document.getElementById("cluster-notes");
            noteBox.value = d.notes || "";
            noteBox.oninput = (ev) => {{ 
                d.notes = ev.target.value; 
                updateAnnotations(); // dynamically render
            }};
            
            // 1. List Features
            const featDiv = document.getElementById("list-features");
            featDiv.innerHTML = "";
            d.features.forEach(f => {{
                const el = document.createElement("div");
                el.className = "list-item";
                el.innerHTML = `
                    <div class="feat-row">
                        <span class="feat-head">Feature ${{f.feat_idx}}</span>
                        <span class="feat-sub">${{f.hook}}</span>
                    </div>
                    <span class="edge-val" style="color:${{f.score>0?'#28a745':'#dc3545'}}">${{f.score.toFixed(3)}}</span>
                `;
                el.onclick = () => showFeature(f);
                featDiv.appendChild(el);
            }});
            
            // 2. Inputs
            updateEdgeList("list-in", l => l.target.id === d.id, l => l.source);
            // 3. Outputs
            updateEdgeList("list-out", l => l.source.id === d.id, l => l.target);
            
            showClusterView(); // Ensure correct view
        }}

        function updateEdgeList(divId, filter, getOther) {{
            const div = document.getElementById(divId);
            div.innerHTML = "";
            const edges = data.links.filter(filter).sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
            
            if(edges.length === 0) div.innerHTML = "<div style='padding:10px;color:#999'>None</div>";
            
            edges.forEach(l => {{
                const other = getOther(l);
                const el = document.createElement("div");
                el.className = "list-item";
                el.innerHTML = `
                    <div class="feat-row">
                        <span class="feat-head">Cluster ${{other.cluster_label}}</span>
                        <span class="feat-sub">${{other.features.length}} nodes</span>
                    </div>
                    <div style="text-align:right">
                         <div class="edge-val ${{l.value>0?'pos':'neg'}}">${{l.value.toFixed(4)}}</div>
                         <div style="font-size:9px;color:#aaa">n=${{l.count}}</div>
                    </div>
                `;
                el.onclick = () => {{
                   // Find node and trigger click
                   const n = d3.selectAll(".node").filter(dn => dn.id === other.id);
                   if(!n.empty()) n.dispatch("click");
                }};
                div.appendChild(el);
            }});
        }}
    }}
    
    // View Switching
    function showClusterView() {{
        document.getElementById("cluster-view").style.display = "block";
        document.getElementById("feature-view").style.display = "none";
    }}
    
    function showFeature(f) {{
        document.getElementById("cluster-view").style.display = "none";
        const view = document.getElementById("feature-view");
        view.style.display = "block";
        
        document.getElementById("feat-title").innerHTML = `Feature ${{f.feat_idx}} <small style='color:#777'>(${{f.hook}})</small>`;
        
        // Handle Feature Notes
        const featNoteBox = document.getElementById("feat-notes");
        featNoteBox.value = f.notes || "";
        featNoteBox.oninput = (e) => {{ f.notes = e.target.value; }};
        
        const content = document.getElementById("feat-content");
        
        // Metadata: Seq, Score
        let metaHtml = `
            <div style="font-size:12px; color:#555; margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid #eee;">
                <b>Seq Pos:</b> ${{f.seq}}<br>
                <b>Seq str:</b> ${{f.seq_str}}<br>
                <b>Score:</b> ${{f.score.toFixed(4)}}
            </div>
        `;

        // Helper to generate list
        const renderLinkList = (list) => {{
            if (!list || list.length === 0) return "<div style='color:#999'>None</div>";
            list.sort((a,b) => Math.abs(b.score) - Math.abs(a.score));
            return list.map(o => {{
                const cColor = o.cluster !== null ? colorScale(o.cluster) : "#ccc";
                const cBadge = o.cluster !== null 
                    ? `<span style="background:${{cColor}}; color:#fff; padding:1px 4px; border-radius:3px; font-size:9px;">C${{o.cluster}}</span>` 
                    : "<span style='color:#999; font-size:9px;'>(No Cluster)</span>";
                
                return `
                    <div class="list-item" style="font-size:11px; padding:4px;">
                        <div class="feat-row">
                            <span>Feature ${{o.feat_idx}} ${{cBadge}}</span>
                        </div>
                        <div class="edge-val ${{o.score>0?'pos':'neg'}}">${{o.score.toFixed(4)}}</div>
                    </div>
                `;
            }}).join('');
        }};

        // Inputs List
        let inHtml = "<h4>Inputs (Fine-grained)</h4><div style='max-height:150px; overflow-y:auto; border-bottom:1px solid #eee; margin-bottom:10px;'>";
        inHtml += renderLinkList(f.incoming);
        inHtml += "</div>";

        // Outgoing List
        let outHtml = "<h4>Outputs (Fine-grained)</h4><div style='max-height:150px; overflow-y:auto; border-bottom:1px solid #eee; margin-bottom:10px;'>";
        outHtml += renderLinkList(f.outgoing);
        outHtml += "</div>";

        content.innerHTML = metaHtml + inHtml + outHtml + `
            <h4>Logits</h4>
            ${{f.logits_html}}
            <h4>Activations</h4>
            ${{f.act_html}}
        `;
    }}
</script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"Aggregated visualization saved to {output_path}")