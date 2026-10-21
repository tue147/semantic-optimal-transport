import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import random
import json
import os

class SAEBlindEvaluator:
    def __init__(self, eval_data, plot_fn, save_file="sae_eval_progress.json"):
        """
        eval_data: dict formatted as { target_feat_idx: { 'Method_A': match_idx, 'Method_B': match_idx, ... } }
        plot_fn: callable that takes a feature index and plots it
        save_file: path to save the JSON progress file
        """
        self.eval_data = eval_data
        self.plot_fn = plot_fn
        self.save_file = save_file
        self.targets = list(eval_data.keys())
        
        self.current_idx = 0
        self.methods = set()
        for matches in eval_data.values():
            self.methods.update(matches.keys())
        self.scores = {m: {'best': 0, 'valid': 0} for m in self.methods}
        
        self._load_progress_from_file()
        
        # State mappings for the current trial
        self.current_mapping = {}     # Option name -> List of methods that predicted it
        self.option_to_feature = {}   # Option name -> Feature ID
        
        self._setup_ui()
        self._load_current()

    def _load_progress_from_file(self):
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r') as f:
                    saved_state = json.load(f)
                    self.scores = saved_state.get('scores', self.scores)
                    self.current_idx = saved_state.get('current_idx', 0)
                print(f"✅ Loaded previous progress. Resuming at target {self.current_idx + 1} / {len(self.targets)}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load save file ({e}). Starting fresh.")

    def _save_progress_to_file(self):
        state = {
            'current_idx': self.current_idx,
            'scores': self.scores
        }
        with open(self.save_file, 'w') as f:
            json.dump(state, f, indent=4)
        
        self.save_status.value = f"💾 Auto-saved at target {self.current_idx}"

    def _setup_ui(self):
        self.progress_label = widgets.Label(style={'font_weight': 'bold'})
        self.save_status = widgets.Label(style={'color': 'gray'}) 
        
        self.target_title = widgets.HTML()
        self.target_output = widgets.Output()
        self.target_box = widgets.VBox([self.target_title, self.target_output], layout=widgets.Layout(align_items='flex-start'))
        
        self.matches_container = widgets.HBox() 
        
        self.valid_checkboxes = []
        self.valid_box = widgets.VBox([])
        
        self.best_checkboxes = []
        self.best_box = widgets.VBox([])
        
        self.next_btn = widgets.Button(description="Submit & Next", button_style='success', icon='check')
        self.next_btn.on_click(self._on_next_clicked)
        
        self.controls_box = widgets.VBox([
            widgets.HTML("<b>Select ALL Valid Matches:</b>"),
            self.valid_box,
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Select BEST Match(es) (Ties allowed):</b>"),
            self.best_box,
            widgets.HTML("<br>"),
            widgets.HBox([self.next_btn, self.save_status]) 
        ])
        
        self.main_layout = widgets.VBox([
            self.progress_label,
            self.target_box,
            widgets.HTML("<hr><h2>Anonymized Matches (Deduplicated)</h2>"),
            self.matches_container,
            widgets.HTML("<hr>"),
            self.controls_box
        ])
        
        display(self.main_layout)

    def _load_current(self):
        if self.current_idx >= len(self.targets):
            self._show_results()
            return
            
        target_feat = self.targets[self.current_idx]
        matches = self.eval_data[target_feat]
        
        self.progress_label.value = f"Evaluating: {self.current_idx + 1} / {len(self.targets)}"
        
        # --- NEW DEDUPLICATION LOGIC ---
        # Group methods by the feature they predicted
        feature_to_methods = {}
        for method, feat in matches.items():
            if feat not in feature_to_methods:
                feature_to_methods[feat] = []
            feature_to_methods[feat].append(method)
            
        # Get unique features and shuffle them
        unique_features = list(feature_to_methods.keys())
        random.shuffle(unique_features)
        
        self.current_mapping = {}
        self.option_to_feature = {}
        
        for i, feat in enumerate(unique_features):
            opt_name = f"Option {i+1}"
            self.current_mapping[opt_name] = feature_to_methods[feat] # List of methods for this option
            self.option_to_feature[opt_name] = feat                   # The actual feature ID
        # --------------------------------
        
        options = list(self.current_mapping.keys())
        
        self.valid_checkboxes = [widgets.Checkbox(value=False, description=opt) for opt in options]
        self.valid_box.children = self.valid_checkboxes
        
        self.best_checkboxes = [widgets.Checkbox(value=False, description=opt) for opt in options]
        self.best_box.children = self.best_checkboxes
        
        plt.close('all')
        
        self.target_title.value = f"<h2>TARGET FEATURE: {target_feat}</h2>"
        with self.target_output:
            clear_output(wait=True)
            with plt.rc_context({'figure.figsize': (3.5, 3.5)}):
                self.plot_fn(target_feat)
                if plt.get_fignums():
                    plt.show()
            
        match_outputs = []
        for opt in options:
            match_feat = self.option_to_feature[opt]
            
            out = widgets.Output()
            with out:
                with plt.rc_context({'figure.figsize': (3.5, 3.5)}):
                    self.plot_fn(match_feat)
                    if plt.get_fignums():
                        plt.show()
            
            label = widgets.HTML(f"<h4 style='text-align: center; margin-bottom: 0px;'>{opt}<br>(Feature {match_feat})</h4>")
            box = widgets.VBox([label, out], layout=widgets.Layout(align_items='center', margin='0px 10px 0px 0px'))
            match_outputs.append(box)
            
        self.matches_container.children = match_outputs

    def _on_next_clicked(self, b):
        # Tally Valid votes for ALL methods that predicted this option
        for cb in self.valid_checkboxes:
            if cb.value:
                methods_list = self.current_mapping[cb.description]
                for method in methods_list:
                    self.scores[method]['valid'] += 1
                
        # Tally Best votes for ALL methods that predicted this option
        for cb in self.best_checkboxes:
            if cb.value:
                methods_list = self.current_mapping[cb.description]
                for method in methods_list:
                    self.scores[method]['best'] += 1
            
        self.current_idx += 1
        self._save_progress_to_file()
        self._load_current()

    def _show_results(self):
        self.main_layout.children = []
        clear_output()
        
        print("=== EVALUATION COMPLETE ===")
        print(f"Total Evaluated: {len(self.targets)}\n")
        print(f"Final results saved to: {self.save_file}\n")
        
        for method, score in self.scores.items():
            best_rate = (score['best'] / len(self.targets)) * 100
            valid_rate = (score['valid'] / len(self.targets)) * 100
            print(f"Method: {method}")
            print(f"  - Chosen as Best:  {score['best']} times ({best_rate:.1f}%)")
            print(f"  - Marked as Valid: {score['valid']} times ({valid_rate:.1f}%)")
            print("-" * 30)