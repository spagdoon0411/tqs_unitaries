from pydantic import BaseModel, Field
from typing import List, Optional


class DiagnosticSetting(BaseModel):
    enabled: bool = True
    exclude_layers: List[str] = Field(default_factory=list)


class ThresholdSetting(DiagnosticSetting):
    threshold: float = 0.0


class DiagnosticConfig(BaseModel):
    global_frequency: int = Field(default=100, ge=1)

    mean_activations: DiagnosticSetting = Field(default_factory=DiagnosticSetting)
    std_activations: DiagnosticSetting = Field(default_factory=DiagnosticSetting)
    activations_below_threshold: ThresholdSetting = Field(
        default_factory=ThresholdSetting
    )
    gradient_l2_norm: DiagnosticSetting = Field(default_factory=DiagnosticSetting)
    weight_l2_norm: DiagnosticSetting = Field(default_factory=DiagnosticSetting)
    update_to_weight_ratio: DiagnosticSetting = Field(default_factory=DiagnosticSetting)
    activation_percentiles: DiagnosticSetting = Field(default_factory=DiagnosticSetting)


# Default configuration
DEFAULT_CONFIG = DiagnosticConfig(
    global_frequency=10,
    mean_activations=DiagnosticSetting(enabled=True, exclude_layers=[]),
    std_activations=DiagnosticSetting(enabled=True, exclude_layers=[]),
    activations_below_threshold=ThresholdSetting(
        enabled=True, threshold=0.01, exclude_layers=[]
    ),
    gradient_l2_norm=DiagnosticSetting(enabled=True, exclude_layers=[]),
    weight_l2_norm=DiagnosticSetting(enabled=True, exclude_layers=[]),
    update_to_weight_ratio=DiagnosticSetting(enabled=True, exclude_layers=[]),
    activation_percentiles=DiagnosticSetting(enabled=True, exclude_layers=[]),
)
