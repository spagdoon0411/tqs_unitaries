import torch
import numpy as np
from typing import Dict, Any, Optional
from diagnostic_config import DiagnosticConfig


class DiagnosticLogger:
    def __init__(self, neptune_run, config: DiagnosticConfig):
        self.neptune_run = neptune_run
        self.config = config
        self.step_count = 0
        self.hooks = []
    
    def register_hooks(self, model: torch.nn.Module):
        """Register forward hooks to capture activations"""
        def hook_fn(module, input, output):
            module._last_activation = output.detach() if isinstance(output, torch.Tensor) else output
        
        for name, module in model.named_modules():
            if not self._should_exclude_layer(name):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def should_log(self) -> bool:
        return self.step_count % self.config.global_frequency == 0
    
    def log_diagnostics(self, model: torch.nn.Module, step: Optional[int] = None):
        if not self.should_log():
            self.step_count += 1
            return
        
        step = step or self.step_count
        diagnostics = {}
        
        for name, module in model.named_modules():
            if self._should_exclude_layer(name):
                continue
                
            # Activations (if available from forward hooks)
            if hasattr(module, '_last_activation'):
                act = module._last_activation
                diagnostics.update(self._compute_activation_stats(name, act))
            
            # Weights
            if hasattr(module, 'weight') and module.weight is not None:
                diagnostics.update(self._compute_weight_stats(name, module.weight))
            
            # Gradients
            if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                diagnostics.update(self._compute_gradient_stats(name, module.weight))
        
        # Log by layer - one plot per metric under each layer
        for key, value in diagnostics.items():
            self.neptune_run[f"diagnostics/by_layer/{key}"].log(value, step=step)
        
        # Log by diagnostic type - one plot per layer under each type
        for name, module in model.named_modules():
            if self._should_exclude_layer(name):
                continue
                
            # Log each layer's metrics under the diagnostic type
            if hasattr(module, '_last_activation'):
                act = module._last_activation
                layer_stats = self._compute_activation_stats(name, act)
                
                for metric_key, value in layer_stats.items():
                    if "/mean_activation" in metric_key and self.config.mean_activations.enabled:
                        self.neptune_run[f"diagnostics/by_type/mean_activations/{name}"].log(value, step=step)
                    elif "/std_activation" in metric_key and self.config.std_activations.enabled:
                        self.neptune_run[f"diagnostics/by_type/std_activations/{name}"].log(value, step=step)
                    elif "/fraction_below_threshold" in metric_key and self.config.activations_below_threshold.enabled:
                        self.neptune_run[f"diagnostics/by_type/activations_below_threshold/{name}"].log(value, step=step)
                    elif "/activation_1st_percentile" in metric_key and self.config.activation_percentiles.enabled:
                        self.neptune_run[f"diagnostics/by_type/activation_1st_percentile/{name}"].log(value, step=step)
                    elif "/activation_99th_percentile" in metric_key and self.config.activation_percentiles.enabled:
                        self.neptune_run[f"diagnostics/by_type/activation_99th_percentile/{name}"].log(value, step=step)
            
            if hasattr(module, 'weight') and module.weight is not None:
                weight_stats = self._compute_weight_stats(name, module.weight)
                for metric_key, value in weight_stats.items():
                    if "/weight_l2_norm" in metric_key and self.config.weight_l2_norm.enabled:
                        self.neptune_run[f"diagnostics/by_type/weight_l2_norm/{name}"].log(value, step=step)
            
            if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                grad_stats = self._compute_gradient_stats(name, module.weight)
                for metric_key, value in grad_stats.items():
                    if "/gradient_l2_norm" in metric_key and self.config.gradient_l2_norm.enabled:
                        self.neptune_run[f"diagnostics/by_type/gradient_l2_norm/{name}"].log(value, step=step)
                    elif "/update_to_weight_ratio" in metric_key and self.config.update_to_weight_ratio.enabled:
                        self.neptune_run[f"diagnostics/by_type/update_to_weight_ratio/{name}"].log(value, step=step)
        
        self.step_count += 1
    
    def _should_exclude_layer(self, layer_name: str) -> bool:
        # Check against all exclude lists
        for setting in [self.config.mean_activations, self.config.std_activations,
                       self.config.activations_below_threshold, self.config.gradient_l2_norm,
                       self.config.weight_l2_norm, self.config.update_to_weight_ratio,
                       self.config.activation_percentiles]:
            if any(exclude in layer_name for exclude in setting.exclude_layers):
                return True
        return False
    
    def _compute_activation_stats(self, layer_name: str, activations) -> Dict[str, float]:
        stats = {}
        
        # Handle tuple outputs - process each tensor separately
        if isinstance(activations, tuple):
            for i, item in enumerate(activations):
                if isinstance(item, torch.Tensor):
                    sub_stats = self._compute_tensor_stats(f"{layer_name}_output_{i}", item)
                    stats.update(sub_stats)
            return stats
        
        # Handle single tensor
        if isinstance(activations, torch.Tensor):
            return self._compute_tensor_stats(layer_name, activations)
        
        return stats
    
    def _compute_tensor_stats(self, layer_name: str, tensor: torch.Tensor) -> Dict[str, float]:
        stats = {}
        act_flat = tensor.flatten().detach().cpu()
        
        if self.config.mean_activations.enabled:
            stats[f"{layer_name}/mean_activation"] = float(act_flat.mean())
        
        if self.config.std_activations.enabled:
            stats[f"{layer_name}/std_activation"] = float(act_flat.std())
        
        if self.config.activations_below_threshold.enabled:
            below_thresh = (act_flat < self.config.activations_below_threshold.threshold).float().mean()
            stats[f"{layer_name}/fraction_below_threshold"] = float(below_thresh)
        
        if self.config.activation_percentiles.enabled:
            percentiles = torch.quantile(act_flat, torch.tensor([0.01, 0.99]))
            stats[f"{layer_name}/activation_1st_percentile"] = float(percentiles[0])
            stats[f"{layer_name}/activation_99th_percentile"] = float(percentiles[1])
        
        return stats
    
    def _compute_weight_stats(self, layer_name: str, weights: torch.Tensor) -> Dict[str, float]:
        stats = {}
        
        if self.config.weight_l2_norm.enabled:
            stats[f"{layer_name}/weight_l2_norm"] = float(torch.norm(weights))
        
        return stats
    
    def _compute_gradient_stats(self, layer_name: str, weights: torch.Tensor) -> Dict[str, float]:
        stats = {}
        
        if self.config.gradient_l2_norm.enabled:
            stats[f"{layer_name}/gradient_l2_norm"] = float(torch.norm(weights.grad))
        
        if self.config.update_to_weight_ratio.enabled and hasattr(weights, '_last_update'):
            update_norm = torch.norm(weights._last_update)
            weight_norm = torch.norm(weights)
            stats[f"{layer_name}/update_to_weight_ratio"] = float(update_norm / (weight_norm + 1e-8))
        
        return stats
