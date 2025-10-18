"""
Training IntegratorLanguageModel LARGE

Entraîner un vrai LLM basé sur INL avec plus de paramètres.

Configurations disponibles:
- Small: ~10M params (comparable GPT-2 small)
- Medium: ~50M params
- Large: ~150M params
"""

import os
# Désactiver le warning des tokenizers lors du fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json

from integrator_language_model import IntegratorLanguageModel


# Configurations de modèles
MODEL_CONFIGS = {
    'small': {
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'num_iterations_per_layer': 5,
        'feedforward_dim': 2048,
        'max_seq_len': 512,
        'params_approx': '~43M'
    },
    'medium': {
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'num_iterations_per_layer': 7,
        'feedforward_dim': 3072,
        'max_seq_len': 1024,
        'params_approx': '~112M'
    },
    'large': {
        'd_model': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'num_iterations_per_layer': 10,
        'feedforward_dim': 4096,
        'max_seq_len': 2048,
        'params_approx': '~303M'
    },
    'xlarge': {
        'd_model': 1536,
        'num_layers': 32,
        'num_heads': 24,
        'num_iterations_per_layer': 12,
        'feedforward_dim': 6144,
        'max_seq_len': 4096,
        'params_approx': '~808M'
    },
    '3b': {
        'd_model': 2048,
        'num_layers': 40,
        'num_heads': 32,
        'num_iterations_per_layer': 15,
        'feedforward_dim': 8192,
        'max_seq_len': 4096,
        'params_approx': '~3B'
    },
    '4b': {
        'd_model': 2304,
        'num_layers': 48,
        'num_heads': 36,
        'num_iterations_per_layer': 15,
        'feedforward_dim': 9216,
        'max_seq_len': 4096,
        'params_approx': '~4B'
    }
}


class TextDataset(Dataset):
    """Dataset pour fichiers texte ou parquet."""

    def __init__(self, file_path, tokenizer, max_length=512, max_samples=None):
        print(f"\nChargement dataset: {file_path}")

        # Détecter le type de fichier
        if file_path.endswith('.parquet'):
            print("  Type: Parquet")
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            df = table.to_pandas()

            # Essayer de trouver les colonnes de texte
            text_column = None
            combine_columns = False

            # Vérifier si c'est un format conversationnel (prompt + completion)
            if 'prompt' in df.columns and 'completion' in df.columns:
                print(f"  Format détecté: Conversationnel (prompt + completion)")
                chunks = []
                for _, row in df.iterrows():
                    # Combiner prompt et completion
                    combined = f"{row['prompt']}\n\n{row['completion']}"
                    if len(combined.strip()) > 50:
                        chunks.append(combined.strip())
                combine_columns = True

            # Sinon, chercher une colonne de texte unique
            if not combine_columns:
                for col in ['content', 'text', 'code', 'data']:
                    if col in df.columns:
                        text_column = col
                        break

                if text_column is None:
                    # Prendre la première colonne de type string
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            text_column = col
                            break

                if text_column is None:
                    raise ValueError(f"Impossible de trouver une colonne de texte dans {file_path}. Colonnes: {df.columns.tolist()}")

                print(f"  Colonne utilisée: '{text_column}'")

                # Extraire les textes
                chunks = df[text_column].astype(str).tolist()
                chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        else:
            # Fichier texte classique
            print("  Type: Text")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split en chunks
            chunks = text.split('\n\n')
            chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        if max_samples:
            chunks = chunks[:max_samples]

        print(f"  Chunks: {len(chunks)}")

        # Tokenize
        self.examples = []
        print("  Tokenization...")
        for chunk in tqdm(chunks):
            encoded = tokenizer.encode(
                chunk,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append(encoded.squeeze(0))

        print(f"  ✓ {len(self.examples)} exemples prêts")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_inl_llm(model_size='small', vocab_size=50257):
    """
    Créer un IntegratorLanguageModel de taille donnée.

    Args:
        model_size: 'small', 'medium', ou 'large'
        vocab_size: Taille du vocabulaire

    Returns:
        model: IntegratorLanguageModel
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Taille inconnue: {model_size}. Choix: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_size]

    print(f"\n{'='*60}")
    print(f"Création IntegratorLanguageModel HYBRIDE ({model_size.upper()})")
    print(f"{'='*60}")
    print(f"  Architecture: Attention + INL")
    print(f"  Paramètres estimés: {config['params_approx']}")
    print(f"  d_model: {config['d_model']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Attention heads: {config['num_heads']}")
    print(f"  INL iterations/layer: {config['num_iterations_per_layer']}")
    print(f"  Max sequence length: {config['max_seq_len']}")

    model = IntegratorLanguageModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_iterations_per_layer=config['num_iterations_per_layer'],
        feedforward_dim=config['feedforward_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=0.1,
        tie_weights=True,
        use_attention=True
    )

    actual_params = model.get_num_params()
    print(f"\n  Paramètres réels: {actual_params:,}")
    print(f"{'='*60}")

    return model


def train_inl_llm(
    data_path,
    model_size='small',
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    max_samples=None,
    save_dir='./checkpoints/inl_llm',
    device=None
):
    """
    Entraîner IntegratorLanguageModel.

    Args:
        data_path: Chemin vers fichier .txt
        model_size: 'small', 'medium', ou 'large'
        num_epochs: Nombre d'epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_samples: Limiter nombre d'exemples (None = tous)
        save_dir: Dossier de sauvegarde
        device: Device (None = auto)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'#'*60}")
    print(f"# TRAINING INL-LLM ({model_size.upper()})")
    print(f"{'#'*60}")
    print(f"\nDevice: {device}")
    print(f"Data: {data_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # Tokenizer
    print("\n" + "="*60)
    print("TOKENIZER")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")

    # Dataset
    print("\n" + "="*60)
    print("DATASET")
    print("="*60)
    max_length = MODEL_CONFIGS[model_size]['max_seq_len']
    dataset = TextDataset(data_path, tokenizer, max_length=max_length, max_samples=max_samples)

    # Train/Val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"  Train: {train_size} exemples ({len(train_loader)} batches)")
    print(f"  Val: {val_size} exemples ({len(val_loader)} batches)")

    # Modèle
    model = create_inl_llm(model_size, vocab_size).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # LR Scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'perplexity': []
    }

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for batch_idx, input_ids in enumerate(pbar):
            input_ids = input_ids.to(device)

            # Forward
            logits, _ = model(input_ids)

            # Loss (next token prediction)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for input_ids in pbar:
                input_ids = input_ids.to(device)

                logits, _ = model(input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss = criterion(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1)
                )

                total_val_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        history['perplexity'].append(perplexity)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f'best_model_{model_size}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'perplexity': perplexity,
                'config': MODEL_CONFIGS[model_size],
                'vocab_size': vocab_size
            }, save_path)
            print(f"  ✓ Best model saved: {save_path}")

        # Sample génération
        if (epoch + 1) % 2 == 0:
            print("\n  Sample génération:")
            prompts = ["Le", "Python est", "L'intelligence artificielle"]

            model.eval()
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

                with torch.no_grad():
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=30,
                        temperature=0.8,
                        top_k=50,
                        do_sample=True
                    )

                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"    '{prompt}' → '{text}'")
            print()

    # Save final
    final_path = os.path.join(save_dir, f'final_model_{model_size}.pt')
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': history['val_loss'][-1],
        'perplexity': history['perplexity'][-1],
        'config': MODEL_CONFIGS[model_size],
        'vocab_size': vocab_size,
        'history': history
    }, final_path)

    # Save history
    with open(os.path.join(save_dir, f'history_{model_size}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("✓ TRAINING TERMINÉ")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final perplexity: {history['perplexity'][-1]:.2f}")
    print(f"  Models saved in: {save_dir}")

    return model, tokenizer, history


def main():
    """Menu principal."""
    print("\n" + "#"*60)
    print("# IntegratorLanguageModel - Large Training")
    print("#"*60)

    # Sélection taille
    print("\nTaille du modèle:")
    for size, config in MODEL_CONFIGS.items():
        print(f"  {size}: {config['params_approx']} params, {config['num_layers']} layers")

    model_size = input("\nChoisir taille (small/medium/large): ").strip().lower()
    if model_size not in MODEL_CONFIGS:
        print("Taille invalide, utilisation de 'small' par défaut")
        model_size = 'small'

    # Data path
    data_path = input("\nChemin vers fichier (.txt ou .parquet): ").strip()
    if not os.path.exists(data_path):
        print(f"❌ Fichier non trouvé: {data_path}")
        print("\nFormats acceptés:")
        print("  - .txt : fichier texte")
        print("  - .parquet : fichier parquet (ex: TheStack)")
        return

    # Params
    num_epochs = int(input("Nombre d'epochs (10): ") or "10")
    batch_size = int(input(f"Batch size ({4 if model_size=='large' else 8}): ") or (4 if model_size=='large' else 8))

    # Launch training
    model, tokenizer, history = train_inl_llm(
        data_path=data_path,
        model_size=model_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=5e-5 if model_size != 'large' else 3e-5,
        save_dir=f'./checkpoints/inl_llm_{model_size}'
    )

    print("\n" + "#"*60)
    print("# Terminé !")
    print("#"*60)
    print("\nVous pouvez maintenant:")
    print(f"  - Charger le modèle depuis ./checkpoints/inl_llm_{model_size}/best_model_{model_size}.pt")
    print("  - Générer du texte avec model.generate()")
    print("  - Fine-tuner sur données spécifiques")


if __name__ == '__main__':
    main()
