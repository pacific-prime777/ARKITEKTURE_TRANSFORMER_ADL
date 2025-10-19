"""
Integrator Neural Language Model (INL-LLM)

A novel language model architecture based on integrator dynamics and learnable equilibrium.

Author: Boris Peyriguère
"""

__version__ = "2.0.0"
__author__ = "Boris Peyriguère"

# Import key components for easy access
from .models.integrator_language_model_optimized import (
    create_optimized_model,
    OptimizedIntegratorLanguageModel
)

from .models.integrator_language_model_ultra import (
    create_ultra_optimized_model,
    UltraOptimizedIntegratorLanguageModel
)

# Alias: ultra model devient le défaut recommandé
create_model = create_ultra_optimized_model  # RECOMMENDED: Maximum efficiency

__all__ = [
    'create_model',  # Recommended default (ultra-optimized)
    'create_optimized_model',  # Level 1 only
    'create_ultra_optimized_model',  # Level 1 + 2 (same as create_model)
    'OptimizedIntegratorLanguageModel',
    'UltraOptimizedIntegratorLanguageModel'
]
