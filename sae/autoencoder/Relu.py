from .Standard import *
import einops
   
class ReluSAE(SAE_Template):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_names: List[str],
        dead_steps_threshold: int,
        target_l0: float = 32,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
        sparsity_coef: float = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            hook_names=hook_names,
            auxk=auxk,
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=dead_threshold,
            standardize=standardize,
        )
        self.target_l0 = target_l0
        self.sparsity_coef = sparsity_coef

    def encode(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, TopK | None, Stats | None, torch.Tensor]:
        hidden_pre, stats = self.get_pre_act(inputs)
        latents = self.hook_sae_acts_post(torch.relu(hidden_pre))
        
        self.update_last_nonzero(latents, inputs.device)
        dead, auxk = self.compute_dead_latents_and_auxk(hidden_pre)

        return latents, auxk, stats, dead

    def decode(self, latents: torch.Tensor, stats: Stats | None = None) -> torch.Tensor:
        recons = (latents @ self.decoder.weight.T) + self.pre_encoder_bias
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return self.hook_sae_recons(recons)
    
    def update_last_nonzero(self, latents: torch.Tensor, device: torch.device) -> None:
        # Update the number of steps since the latents have activated
        self.last_nonzero *= (
            1 - 
            (latents > self.cfg.dead_threshold).to(self.last_nonzero.dtype).sum(dim=tuple(range(0, latents.ndim-1))).clamp(max=1)
        )
        self.last_nonzero += 1
        
    def compute_dead_latents_and_auxk(self, latents: Tensor) -> tuple[Tensor, TopK | None]:
        # Mask the latents flagged as dead during training
        dead_mask = self.last_nonzero >= self.cfg.dead_steps_threshold
        latents = latents * dead_mask

        # Compute the fraction of dead latents
        dead = torch.sum(dead_mask, dtype=torch.float32).detach() / self.cfg.d_sae

        # If auxk is not None, find the auxk largest dead latents
        auxk = None
        if self.cfg.auxk is not None:
            values_auxk , indices_auxk = torch.topk(
                latents, 
                k=self.cfg.auxk, 
                sorted=False
            )
            auxk = TopK(values_auxk , indices_auxk)
        return dead, auxk

    def forward_training(self, inputs: torch.Tensor) -> SAEOut:
        latents, auxk, stats, dead = self.encode(inputs)

        recons = self.decode(latents, stats)
        
        # sparsity_loss = latents.sum(dim=-1).mean() * self.sparsity_coef
        
        l0 = (latents > self.cfg.dead_threshold).sum(dim=-1).mean().detach()
        l1 = latents.sum(dim=-1).mean()
        sparsity_loss = (
            self.sparsity_coef * ((l0 / self.target_l0) - 1) * l1
        )

        recons = self.hook_sae_output(recons)
        
        auxk_recons = None
        if auxk is not None:
            auxk_latents = torch.zeros_like(latents)
            auxk_latents.scatter_(
                -1, 
                auxk.indices,
                torch.relu(auxk.values), 
            )
            auxk_recons = self.decode(auxk_latents)
            
        return SAEOut(latents, recons, auxk, auxk_recons, dead, sparsity_loss)