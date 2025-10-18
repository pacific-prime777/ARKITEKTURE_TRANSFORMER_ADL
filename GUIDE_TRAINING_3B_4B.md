# Guide Complet : Entraîner INL-LLM 3B-4B

## 🎯 Modèles Disponibles

| Taille | Paramètres Réels | d_model | Layers | VRAM (FP16) | Tokens/sec (A100) |
|--------|------------------|---------|--------|-------------|-------------------|
| **3B** | **1.70B** | 2048 | 40 | **~3.4 GB** | ~1000 |
| **4B** | **2.52B** | 2304 | 48 | **~5.0 GB** | ~800 |

## 📋 Prérequis

### Matériel Minimum

**Pour 3B** :
- GPU : NVIDIA RTX 3090 (24GB) ou A100 (40GB)
- RAM : 32 GB
- Stockage : 100 GB SSD

**Pour 4B** :
- GPU : NVIDIA A100 (40GB) ou mieux
- RAM : 64 GB
- Stockage : 200 GB SSD

### Logiciels

```bash
# Python 3.8+
python3 --version

# PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers
pip install transformers

# Optionnel (pour DeepSpeed)
pip install deepspeed
```

## 📊 Données d'Entraînement

### Quantité Recommandée

| Modèle | Tokens Minimum | Tokens Optimal | Exemples |
|--------|----------------|----------------|----------|
| 3B | 10B tokens | 50-100B tokens | The Stack (code), C4 (texte) |
| 4B | 15B tokens | 100-200B tokens | The Pile, RedPajama |

### Format Attendu

**Option A : Fichier texte simple**
```
data/
└── corpus.txt  # Un grand fichier texte (français, code, etc.)
```

**Option B : Multiple fichiers**
```
data/
├── french_text.txt
├── python_code.txt
├── documentation.txt
└── ...
```

**Option C : Dataset Hugging Face**
```python
from datasets import load_dataset
dataset = load_dataset("bigcode/the-stack", data_dir="data/python")
```

### Exemple : Préparer The Stack (Code Python)

```bash
# 1. Télécharger
git clone https://huggingface.co/datasets/bigcode/the-stack

# 2. Extraire Python
python3 << EOF
from datasets import load_dataset

dataset = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",
    split="train"
)

# Sauvegarder en fichier texte
with open('python_code_corpus.txt', 'w') as f:
    for example in dataset:
        f.write(example['content'])
        f.write('\n\n')  # Séparateur
EOF
```

## 🚀 Entraînement - Commande Simple

### Modèle 3B

```bash
cd /home/boris/vAgent/architecture

python3 train_inl_llm_large.py
# Entrées interactives:
#   Taille: 3b
#   Fichier: /path/to/corpus.txt
#   Epochs: 10
#   Batch size: 4
```

**Temps estimé** :
- 10B tokens, GPU A100 : ~24 heures
- 50B tokens, GPU A100 : ~5 jours

### Modèle 4B

```bash
python3 train_inl_llm_large.py
# Entrées:
#   Taille: 4b
#   Fichier: /path/to/corpus.txt
#   Epochs: 10
#   Batch size: 2  # Plus petit car plus gros modèle
```

**Temps estimé** :
- 10B tokens, GPU A100 : ~36 heures
- 100B tokens, GPU A100 : ~2 semaines

## ⚙️ Entraînement Avancé (Programmation)

### Script Python Custom

```python
from train_inl_llm_large import train_inl_llm

# Entraîner 3B
model, tokenizer, history = train_inl_llm(
    data_path='/path/to/huge_corpus.txt',
    model_size='3b',
    num_epochs=10,
    batch_size=4,              # Ajuster selon VRAM
    learning_rate=3e-5,         # Important pour gros modèles
    max_samples=None,           # Tous les samples
    save_dir='./checkpoints/inl_3b',
    device='cuda'
)
```

### Avec Gradient Accumulation (pour + gros batch)

```python
# Modifier train_inl_llm_large.py :

# Au lieu de:
loss.backward()
optimizer.step()

# Faire:
ACCUMULATION_STEPS = 8
loss = loss / ACCUMULATION_STEPS
loss.backward()

if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
    optimizer.step()
    optimizer.zero_grad()

# Effective batch size = batch_size * ACCUMULATION_STEPS
# Ex: batch_size=2, accum=8 → effective=16
```

### Avec Mixed Precision (FP16/BF16)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # FP16
        logits, _ = model(input_ids)
        loss = criterion(logits, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Avantages** :
- 2x moins de VRAM
- 2x plus rapide
- Même performance finale

## 🔧 Optimisations Recommandées

### 1. DeepSpeed ZeRO Stage 2

```bash
pip install deepspeed

# Créer config DeepSpeed
cat > ds_config.json << EOF
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
EOF

# Lancer avec DeepSpeed
deepspeed train_inl_llm_large.py --deepspeed ds_config.json
```

**Bénéfices** :
- Stage 2 : Partage optimizer states → économie VRAM
- Offload CPU : Utilise RAM pour optimizer
- Peut entraîner 3B sur 16GB VRAM !

### 2. Flash Attention (si implémenté)

```python
# Dans integrator_language_model.py
# Remplacer context projection par Flash Attention

from flash_attn import flash_attn_func

# Au lieu de simple projection
context = flash_attn_func(q, k, v)  # 5-10x plus rapide
```

### 3. Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

# Dans IntegratorLanguageBlock.forward()
def forward(self, x):
    # Normale
    x = self.layer1(x)

    # Avec checkpointing (économise VRAM)
    x = checkpoint(self.layer1, x, use_reentrant=False)
```

**Trade-off** :
- ✅ VRAM : -40%
- ❌ Vitesse : -20%

## 📈 Monitoring et Debugging

### Tracker Métriques

```python
# Ajouter à train_inl_llm_large.py

import wandb

wandb.init(project='inl-llm-3b', config={
    'model_size': '3b',
    'd_model': 2048,
    'num_layers': 40,
    'batch_size': 4,
    'lr': 3e-5
})

# Dans training loop
wandb.log({
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
    'perplexity': perplexity,
    'learning_rate': scheduler.get_last_lr()[0],
    'epoch': epoch
})
```

### Vérifier Utilisation GPU

```bash
# Pendant training
watch -n 1 nvidia-smi

# Devrait voir:
# - GPU Util: ~95-100%
# - Memory: ~90-95% du total
# - Temperature: <85°C
```

### Logs Importants

```python
# Ajouter prints détaillés
print(f"Epoch {epoch}:")
print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
print(f"  Gradient norm: {total_norm:.2f}")
print(f"  Time/batch: {time_per_batch:.2f}s")
print(f"  Tokens/sec: {tokens_per_sec:.0f}")
```

## 🎓 Stratégies d'Entraînement

### Curriculum Learning

```python
# Phase 1: Données simples (50% du temps)
train_on('simple_french.txt', epochs=5)

# Phase 2: Données complexes (30%)
train_on('technical_docs.txt', epochs=3)

# Phase 3: Mix (20%)
train_on('all_data.txt', epochs=2)
```

### Learning Rate Schedule

```python
# Warmup + Cosine Decay
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return current_step / warmup_steps
    else:
        # Cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)
```

### Early Stopping

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

## 🧪 Validation

### Métriques à Suivre

1. **Perplexity** (principal)
   - Cible : < 20 pour texte
   - < 10 pour code simple

2. **Train/Val Loss**
   - Gap < 0.5 : bon
   - Gap > 1.0 : overfitting

3. **Génération Qualité**
   - Cohérence sémantique
   - Grammaire correcte
   - Suit le prompt

### Tests de Génération

```python
# Tous les 5 epochs
test_prompts = [
    "def fibonacci(n):",
    "La capitale de la France est",
    "Pour implémenter un réseau de neurones"
]

for prompt in test_prompts:
    generated = model.generate(...)
    print(f"{prompt} → {generated}")
```

## 💾 Sauvegarde et Reprise

### Checkpoints

```python
# Sauvegarder régulièrement
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'history': history
    }, f'checkpoint_epoch_{epoch}.pt')
```

### Reprise Training

```python
# Charger checkpoint
checkpoint = torch.load('checkpoint_epoch_20.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1

# Continuer
for epoch in range(start_epoch, num_epochs):
    ...
```

## 📊 Estimation Coûts

### Cloud GPU (Vast.ai, RunPod, etc.)

| GPU | VRAM | Prix/h | 3B (50B tokens) | 4B (100B tokens) |
|-----|------|--------|-----------------|------------------|
| RTX 4090 | 24GB | $0.30 | $36 (5j) | ❌ Pas assez VRAM |
| A100 40GB | 40GB | $1.00 | $120 (5j) | $336 (14j) |
| A100 80GB | 80GB | $1.50 | $180 (5j) | $504 (14j) |

### Local (électricité uniquement)

- GPU A100 : ~300W
- 5 jours × 24h × 300W = 36 kWh
- Coût : ~3-5€ (selon pays)

## ✅ Checklist Avant de Lancer

- [ ] GPU avec assez de VRAM (voir tableau)
- [ ] Données préparées (>10B tokens)
- [ ] PyTorch + CUDA installés
- [ ] Espace disque suffisant (200GB+)
- [ ] Mixed precision activée (FP16/BF16)
- [ ] Monitoring configuré (wandb/tensorboard)
- [ ] Checkpointing automatique
- [ ] Tests de génération préparés

## 🚀 Commande Finale Recommandée

### Pour 3B (GPU 24GB+)

```bash
python3 << 'ENDPY'
from train_inl_llm_large import train_inl_llm

model, tokenizer, history = train_inl_llm(
    data_path='/path/to/50B_tokens_corpus.txt',
    model_size='3b',
    num_epochs=10,
    batch_size=4,
    learning_rate=3e-5,
    save_dir='./checkpoints/inl_3b_production'
)
ENDPY
```

### Pour 4B (GPU 40GB+)

```bash
python3 << 'ENDPY'
from train_inl_llm_large import train_inl_llm

model, tokenizer, history = train_inl_llm(
    data_path='/path/to/100B_tokens_corpus.txt',
    model_size='4b',
    num_epochs=10,
    batch_size=2,
    learning_rate=2e-5,
    save_dir='./checkpoints/inl_4b_production'
)
ENDPY
```

## 📚 Ressources Utiles

### Datasets Recommandés

- **The Stack** : 3TB de code (tous langages)
- **The Pile** : 800GB de texte anglais
- **mC4** : Texte multilingue (dont français)
- **RedPajama** : 1.2TB open source

### Outils

- **Hugging Face Hub** : Charger datasets
- **DeepSpeed** : Optimiser training
- **Weights & Biases** : Monitoring
- **TensorBoard** : Visualisation

---

**BON ENTRAÎNEMENT ! 🚀**

Questions ? Consultez README_LLM_INL.md ou ouvrez une issue.
