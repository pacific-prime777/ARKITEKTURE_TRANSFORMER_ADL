# INL-LLM Checkpoints

Ce dossier contient les modèles entraînés avec leur tokenizer.

## Structure du checkpoint

Après l'entraînement avec `simple_training.py`, vous aurez:

```
checkpoints/inl_1b_model/
├── pytorch_model.pt          # Poids du modèle (970M paramètres)
├── vocab.json                # Vocabulaire GPT-2 (50257 tokens)
├── merges.txt                # Règles BPE tokenizer
├── tokenizer.json            # Configuration du tokenizer
└── tokenizer_config.json     # Paramètres du tokenizer
```

## Utilisation

### 1. Charger le modèle complet (modèle + tokenizer)

```python
from transformers import AutoTokenizer
from inl_llm.models import UltraOptimizedIntegratorLanguageModel
import torch

# Charger tokenizer
tokenizer = AutoTokenizer.from_pretrained('checkpoints/inl_1b_model')

# Créer architecture
model = UltraOptimizedIntegratorLanguageModel(
    vocab_size=tokenizer.vocab_size,
    d_model=1600,
    num_layers=28,
    num_heads=25,
    num_iterations_per_layer=10,
    feedforward_dim=6400,
    max_seq_len=2048,
    use_lowrank_embeddings=True,
    lowrank_ratio=0.125,
    use_shared_controllers=True,
    hierarchical_group_size=64,
    excitation_sparsity=0.1
)

# Charger poids
state_dict = torch.load('checkpoints/inl_1b_model/pytorch_model.pt')
model.load_state_dict(state_dict)
model.eval()
```

### 2. Générer du texte

```python
# Tokenize prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# Decode
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)
```

### 3. Script prêt à l'emploi

```bash
python load_and_generate.py
```

## Pourquoi sauvegarder le tokenizer?

Le **tokenizer** est essentiel car:

1. **Texte → Tokens**: Convertit "Hello" en [15496]
2. **Tokens → Texte**: Convertit [15496] en "Hello"
3. **Vocabulaire spécifique**: GPT-2 BPE avec 50257 tokens
4. **Compatibilité**: Sans le bon tokenizer, génération impossible

**Règle d'or**: Toujours sauvegarder modèle + tokenizer ensemble!

## Format vs Llama/GPT

Différence avec les modèles Hugging Face standard:

| Format | Structure |
|--------|-----------|
| **Llama/GPT** | `model.safetensors` ou `pytorch_model.bin` |
| **INL-LLM** | `pytorch_model.pt` (state_dict uniquement) |

Notre format est plus simple car on sauvegarde juste les poids, pas toute la config.
