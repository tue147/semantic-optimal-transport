# Adapted from https://github.com/jbloomAus/SAELens/blob/main/sae_lens/analysis/hooked_sae_transformer.py 

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint, HookedRootModule
from sae_lens import SAE, ActivationsStore

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]


def get_deep_attr(obj: Any, path: str):
    """Helper function to get a nested attribute from a object.
    In practice used to access HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z)

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")

    returns:
        Any. The attribute at the end of the path
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def set_deep_attr(obj: Any, path: str, value: Any):
    """Helper function to change the value of a nested attribute from a object.
    In practice used to swap HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z) with HookedSAEs and vice versa

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")
        value: Any. The value you want to set the attribute to (eg a HookedSAE object)
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    # Set the value on the final attribute
    setattr(obj, parts[-1], value)

def _preprocess_hook_name(child_name: str, parent_name: str):
    assert parent_name + "." in child_name, "Parent class's hook name is not compatible with child class's hook name."
    return child_name.replace(parent_name + ".", "")

class _HookedSAE(HookedRootModule):
    def __init__(
        self,
        *model_args: Any,
        **model_kwargs: Any,
    ):
        """Model initialization. Just SAE init, but adds a dictionary to keep track of attached extending SAEs.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            *model_args: Positional arguments for SAE initialization
            **model_kwargs: Keyword arguments for SAE initialization
        """
        super().__init__(*model_args, **model_kwargs)
        self.acts_to_saes: Dict[str, Any] = {}  # type: ignore
        self.detach_error_term = True
        self.disable_error_grad = False

    def add_sae(self, sae: Any, use_error_term: Optional[bool] = None):
        """Attaches an SAE to the model

        WARNING: This sae will be permanantly attached until you remove it with reset_saes. This function will also overwrite any existing SAE attached to the same hook point.

        Args:
            sae: SparseAutoencoderBase. The SAE to attach to the model
            use_error_term: (Optional[bool]) If provided, will set the use_error_term attribute of the SAE to this value. Determines whether the SAE returns input or reconstruction. Defaults to None.
        """
        act_name = _preprocess_hook_name(sae.cfg.hook_name, self.cfg.hook_name)
        if (act_name not in self.acts_to_saes) and (act_name not in self.hook_dict):
            logging.warning(
                f"No hook found for {act_name}. Skipping. Check model.hook_dict for available hooks."
            )
            return

        if use_error_term is not None:
            if not hasattr(sae, "_original_use_error_term"):
                sae._original_use_error_term = sae.use_error_term  # type: ignore
            sae.use_error_term = use_error_term
        self.acts_to_saes[act_name] = sae
        set_deep_attr(self, act_name, sae)
        self.setup()
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Modify the forward pass to allow gradient flows through the error term.
        '''
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad() if self.detach_error_term else torch.enable_grad():
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                    
            if self.disable_error_grad:
                # If disable_error_grad -> gradient will NOT flows through the error hook
                # Similar to SAE_LENS forward function
                with torch.no_grad():
                    sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            else:
                # If not disable_error_grad -> gradient will flows through the error hook
                # If detach_error_term -> gradient of error term will not affect upstream components
                with torch.no_grad() if self.detach_error_term else torch.enable_grad():
                    temp_error = (x - x_reconstruct_clean)
                temp_error.requires_grad_() # allow gradient flows through the error hook
                
                sae_error = self.hook_sae_error(temp_error)
            sae_out = sae_out + sae_error
            
        return self.hook_sae_output(sae_out)

    def _reset_sae(self, act_name: str, prev_sae: Optional[Any] = None):
        """Resets an SAE that was attached to the model

        By default will remove the SAE from that hook_point.
        If prev_sae is provided, will replace the current SAE with the provided one.
        This is mainly used to restore previously attached SAEs after temporarily running with different SAEs (eg with run_with_saes)

        Args:
            act_name: str. The hook_name of the SAE to reset
            prev_sae: Optional[HookedSAE]. The SAE to replace the current one with. If None, will just remove the SAE from this hook point. Defaults to None
        """
        if act_name not in self.acts_to_saes:
            logging.warning(
                f"No SAE is attached to {act_name}. There's nothing to reset."
            )
            return

        current_sae = self.acts_to_saes[act_name]
        if hasattr(current_sae, "_original_use_error_term"):
            current_sae.use_error_term = current_sae._original_use_error_term  # type: ignore
            delattr(current_sae, "_original_use_error_term")

        if prev_sae:
            set_deep_attr(self, act_name, prev_sae)
            self.acts_to_saes[act_name] = prev_sae
        else:
            set_deep_attr(self, act_name, HookPoint())
            del self.acts_to_saes[act_name]

    def reset_saes(
        self,
        act_names: Optional[Union[str, List[str]]] = None,
        prev_saes: Optional[List[Union[Any, None]]] = None,
    ):
        """Reset the SAEs attached to the model

        If act_names are provided will just reset SAEs attached to those hooks. Otherwise will reset all SAEs attached to the model.
        Optionally can provide a list of prev_saes to reset to. This is mainly used to restore previously attached SAEs after temporarily running with different SAEs (eg with run_with_saes).

        Args:
            act_names (Optional[Union[str, List[str]]): The act_names of the SAEs to reset. If None, will reset all SAEs attached to the model. Defaults to None.
            prev_saes (Optional[List[Union[HookedSAE, None]]]): List of SAEs to replace the current ones with. If None, will just remove the SAEs. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]
        elif act_names is None:
            act_names = list(self.acts_to_saes.keys())

        if prev_saes:
            if len(act_names) != len(prev_saes):
                raise ValueError("act_names and prev_saes must have the same length")
        else:
            prev_saes = [None] * len(act_names)  # type: ignore

        for act_name, prev_sae in zip(act_names, prev_saes):  # type: ignore
            self._reset_sae(act_name, prev_sae)

        self.setup()

    def run_with_saes(
        self,
        *model_args: Any,
        saes: Union[Any, List[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: Optional[bool] = None,
        **model_kwargs: Any,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Wrapper around HookedTransformer forward pass.

        Runs the model with the given SAEs attached for one forward pass, then removes them. By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (Union[HookedSAE, List[HookedSAE]]) The SAEs to be attached for this forward pass
            reset_saes_end (bool): If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. Default is True.
            use_error_term: (Optional[bool]) If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Defaults to None.
            **model_kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self(*model_args, **model_kwargs)

    def run_with_cache_with_saes(
        self,
        *model_args: Any,
        saes: Union[Any, List[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: Optional[bool] = None,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationsStore, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around 'run_with_cache' in HookedTransformer.

        Attaches given SAEs before running the model with cache and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (Union[HookedSAE, List[HookedSAE]]) The SAEs to be attached for this forward pass
            reset_saes_end: (bool) If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. Default is True.
            use_error_term: (Optional[bool]) If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Determines whether the SAE returns input or reconstruction. Defaults to None.
            return_cache_object: (bool) if True, this will return an ActivationCache object, with a bunch of
                useful HookedTransformer specific methods, otherwise it will return a dictionary of
                activations as in HookedRootModule.
            remove_batch_dim: (bool) Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            **kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self.run_with_cache(  # type: ignore
                *model_args,
                remove_batch_dim=remove_batch_dim,
                **kwargs,
            )

    def run_with_hooks_with_saes(
        self,
        *model_args: Any,
        saes: Union[Any, List[Any]] = [],
        reset_saes_end: bool = True,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],  # type: ignore
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],  # type: ignore
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any,
    ):
        """Wrapper around 'run_with_hooks' in HookedTransformer.

        Attaches the given SAEs to the model before running the model with hooks and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            act_names: (Union[HookedSAE, List[HookedSAE]]) The SAEs to be attached for this forward pass
            reset_saes_end: (bool) If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. (default: True)
            fwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of forward hooks to apply
            bwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of backward hooks to apply
            reset_hooks_end: (bool) Whether to reset the hooks at the end of the forward pass (default: True)
            clear_contexts: (bool) Whether to clear the contexts at the end of the forward pass (default: False)
            **model_kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(saes=saes, reset_saes_end=reset_saes_end):
            return self.run_with_hooks(
                *model_args,
                fwd_hooks=fwd_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
                **model_kwargs,
            )

    @contextmanager
    def saes(
        self,
        saes: Union[Any, List[Any]] = [],
        reset_saes_end: bool = True,
        use_error_term: Optional[bool] = None,
    ):
        """
        A context manager for adding temporary SAEs to the model.
        See HookedTransformer.hooks for a similar context manager for hooks.
        By default will keep track of previously attached SAEs, and restore them when the context manager exits.

        Example:

        .. code-block:: python

            from transformer_lens import HookedSAETransformer, HookedSAE, HookedSAEConfig

            model = HookedSAETransformer.from_pretrained('gpt2-small')
            sae_cfg = HookedSAEConfig(...)
            sae = HookedSAE(sae_cfg)
            with model.saes(saes=[sae]):
                spliced_logits = model(text)


        Args:
            saes (Union[HookedSAE, List[HookedSAE]]): SAEs to be attached.
            reset_saes_end (bool): If True, removes all SAEs added by this context manager when the context manager exits, returning previously attached SAEs to their original state.
            use_error_term (Optional[bool]): If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Defaults to None.
        """
        act_names_to_reset = []
        prev_saes = []
        if not isinstance(saes, List):
            saes = [saes]
        try:
            for sae in saes:
                act_names_to_reset.append(_preprocess_hook_name(sae.cfg.hook_name, self.cfg.hook_name))
                prev_sae = self.acts_to_saes.get(_preprocess_hook_name(sae.cfg.hook_name, self.cfg.hook_name), None)
                prev_saes.append(prev_sae)
                self.add_sae(sae, use_error_term=use_error_term)
            yield self
        finally:
            if reset_saes_end:
                self.reset_saes(act_names_to_reset, prev_saes)
   
                
_blank_hook = torch.nn.Identity()

@contextmanager
def _disable_hooks(sae: HookedRootModule):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)
            
class HookedSAE(_HookedSAE, SAE):
    def __init__(
        self,
        *model_args: Any,
        **model_kwargs: Any,
    ):
        """Model initialization. Just SAE init, but adds a dictionary to keep track of attached extending SAEs.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            *model_args: Positional arguments for SAE initialization
            **model_kwargs: Keyword arguments for SAE initialization
        """
        super().__init__(*model_args, **model_kwargs)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(x)