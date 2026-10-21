'''
Adapted from: 
https://github.com/chasewalker26/Integrated-Decision-Gradients/blob/main/util/attribution_methods/saliencyMethods.py
'''

import torch as t
from torch import Tensor
from typing import List, Tuple, Dict, Union, Callable, Any, Iterator
from .path_template import IGPathTemplate

class IDGPath(IGPathTemplate):
    def __init__(
        self,
        target_name: str,
        clean_act: Tensor,
        corrupt_act: Tensor,
        clean_token: Tensor,
        fwd_func: Callable[[Tensor, Dict[str, Tensor], Callable[[Tensor], Tensor]], Dict[str, Tensor]],
        grad_func: Callable[[Tensor, Dict[str, Tensor], Callable[[Tensor], Tensor]], Dict[str, Tensor]],
        metric: Callable[[Tensor], Tensor],
        n_steps: int = 10,
        **kwargs
    ):
        super().__init__()
        self.target_name = target_name
        self.clean_act = clean_act
        self.corrupt_act = corrupt_act
        self.clean_token = clean_token
        self.fwd_func = fwd_func
        self.grad_func = grad_func
        self.metric = metric
        self.n_steps = n_steps
        self.act_path, self.scalers = self.IDG()

    @t.no_grad()
    def IDG(self):
        device = self.clean_act.device
        baseline = self.corrupt_act
        input = self.clean_act
        
        baseline_diff = t.sub(input, baseline) # (b, s, d_sae) | (b, s, d_model)

        # find IG slopes
        slopes, step_size = self.getSlopes(baseline, baseline_diff) # (n_steps), float
        # find new alpha spacing
        alphas, alpha_substep_size = self.getAlphaParameters(slopes, self.n_steps, step_size) # (n_steps), (n_steps)

        alphas = alphas.reshape(self.n_steps).to(device)
        alpha_substep_size = alpha_substep_size.reshape(self.n_steps).to(device)

        # array to store the gradient at each step
        act_paths = []
        # array to store the logit at each step
        logits = t.zeros(self.n_steps).to(device)
        # array to store the slope at each step
        slopes = t.zeros(self.n_steps).to(device)
        
        for i in range(self.n_steps):
            interpolated_input = t.add(baseline, t.mul(alphas[i], baseline_diff))
            act_paths.append(interpolated_input.detach())
            logits[i] = self.getPredictionParallel(interpolated_input)

        # calculate logit slopes
        slopes = t.zeros(self.n_steps).to(device)
        slopes[0] = 0

        # calculate all slopes
        for i in range(0, self.n_steps - 1):
            slopes[i + 1] = (logits[i + 1] - logits[i]) / (alphas[i + 1] - alphas[i])

        scalers = t.multiply(alpha_substep_size, slopes).reshape(self.n_steps)

        return act_paths, scalers # [b, s, d], (n_steps)
    
    @t.no_grad()
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for path, scale in zip(self.act_path, self.scalers):
            yield (path, scale * self.n_steps) # act path and scaler, we will divide by n_steps in the circuit

    # returns the gradients from the grad_func for an input
    def getGradientsParallel(self, patched_act: Tensor) -> Tuple[Tensor, Tensor]:
        output_dict = self.grad_func(
            self.clean_token,
            {self.target_name: patched_act},
            self.metric,
        )
        scores = output_dict["metric"]
        gradients = output_dict[self.target_name]

        return gradients, scores.detach().squeeze()
    
    # returns the output from the fwd_func for an input
    def getPredictionParallel(self, patched_act: Tensor) -> Tensor:
        output_dict = self.fwd_func(
            self.clean_token,
            {self.target_name: patched_act},
            self.metric,
        )
        scores = output_dict["metric"]

        return scores.detach().squeeze()

    def getSlopes(self, baseline: Tensor, baseline_diff: Tensor) -> Tuple[Tensor, float]:
        device = baseline.device

        # generate alpha values as 4D
        alphas = t.linspace(0, 1, self.n_steps)
        alphas = alphas.reshape(self.n_steps).to(device)

        # array to store the scores at each step
        scores = t.zeros(self.n_steps).to(device)

        # run batched input
        for i in range(self.n_steps):
            interp_imgs = t.add(baseline, t.mul(alphas[i], baseline_diff))
            
            scores[i] = self.getPredictionParallel(interp_imgs)

        # calculate logit slopes
        slopes = t.zeros(self.n_steps).to(device)
        x_diff = float(alphas.squeeze()[1] - alphas.squeeze()[0])

        slopes[0] = 0

        # calculate all slopes
        for i in range(0, self.n_steps - 1):
            y_diff = scores[i + 1] - scores[i]
            slopes[i + 1] = y_diff / x_diff

        return slopes, x_diff

    # does an initial point to point slope calculation using a psuedo IG run with steps_hyper steps
    # returns the alpha values to be used as well as the spacing of the alpha values
    def getAlphaParameters(self, slopes: Tensor, steps: int, step_size: float) -> Tuple[Tensor, Tensor]:
        # normalize slopes 0 to 1 to eliminate negatives and preserve magnitude
        slopes_0_1_norm = (slopes - t.min(slopes)) / (t.max(slopes) - t.min(slopes))
        # reset the first slope to zero after normalization because it is impossible to be nonzero
        slopes_0_1_norm[0] = 0
        # normalize the slope values so that they sum to 1.0 and preserve magnitude
        slopes_sum_1_norm = slopes_0_1_norm / t.sum(slopes_0_1_norm)

        # obtain the samples at each alpha step as a float based on the slope (steps/alpha)
        sample_placements_float = t.mul(slopes_sum_1_norm, steps)
        # truncate the result to int values to clean up decimals, this leaves unused steps (samples)
        sample_placements_int = sample_placements_float.type(t.int)
        # find how many unused steps are left
        remaining_to_fill = steps - t.sum(sample_placements_int)

        # find the values which were not truncated to 0 (float values >= 1) 
        # by the int casting and make them -1 in the float array
        non_zeros = t.where(sample_placements_int != 0)[0]
        sample_placements_float[non_zeros] = -1

        # Find the indicies of the remaining spots to fill from the float array (the zero values) sorted high to low
        remaining_hi_lo = t.flip(t.sort(sample_placements_float)[1], dims = [0])
        # Fill all of these spots in the int array with 1, this gives the final distribution of steps
        sample_placements_int[remaining_hi_lo[0 : remaining_to_fill]] = 1

        # holds new alpha values to be created
        alphas = t.zeros(steps)    
        # an array that tracks indivdual steps between alpha values
        # this is important to counteract the non-uniform alpha spacing of this method
        alpha_substep_size = t.zeros(steps)

        # the index at which a range of samples begins, it is a function of num_samples in loop
        alpha_start_index = 0
        # the value at which a range of samples starts, it is a function of step_size
        alpha_start_value = 0

        # generate the new alpha values
        for num_samples in sample_placements_int:        
            if num_samples == 0:
                continue

            # Linearly divide the samples into the required alpha range
            alphas[alpha_start_index: (alpha_start_index + num_samples)] = t.linspace(alpha_start_value, alpha_start_value + step_size, num_samples + 1)[0 : num_samples] # type: ignore

            # track the step size of the alpha divisions
            alpha_substep_size[alpha_start_index: (alpha_start_index + num_samples)] = (step_size / num_samples)

            alpha_start_index += num_samples
            alpha_start_value += step_size

        return alphas, alpha_substep_size