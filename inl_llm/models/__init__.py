"""
Complete INL-LLM models with optimizations.

Models:
- OptimizedIntegratorLanguageModel: Level 1 optimizations (production-ready)
- UltraOptimizedIntegratorLanguageModel: Level 1+2 optimizations (maximum efficiency)
"""

from .integrator_language_model_optimized import (
    OptimizedIntegratorLanguageModel,
    create_optimized_model
)

from .integrator_language_model_ultra import (
    UltraOptimizedIntegratorLanguageModel,
    create_ultra_optimized_model
)

__all__ = [
    'OptimizedIntegratorLanguageModel',
    'create_optimized_model',
    'UltraOptimizedIntegratorLanguageModel',
    'create_ultra_optimized_model'
]
