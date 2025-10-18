"""
IntegratorNeuronLayer Language Model (INL-LM)

Architecture de modèle de langage basée sur la dynamique intégrateur/vitesse
au lieu de l'attention standard des Transformers.

Idée clé : Utiliser la dynamique INL pour modéliser l'évolution
des représentations de tokens au fil du contexte.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

from integrator_neuron_layer import IntegratorNeuronLayer


class PositionalEncoding(nn.Module):
    """Encodage positionnel pour séquences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Créer encodage positionnel
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class IntegratorLanguageBlock(nn.Module):
    """
    Bloc de langage hybride: Attention + INL.

    Architecture améliorée qui combine:
    1. Multi-head attention pour capturer les dépendances contextuelles
    2. Dynamique intégrateur INL pour raffiner les représentations

    Intuition :
    - L'attention extrait le contexte sélectif (comme Transformer)
    - INL applique une dynamique d'équilibre pour stabiliser/raffiner
    - Chaque token a un état x_t et une vitesse v_t
    - Les contrôleurs α, β, g adaptent la dynamique selon le contexte
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_iterations: int = 5,
        target_value: float = 0.0,
        feedforward_dim: int = None,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Args:
            d_model: Dimension du modèle
            num_heads: Nombre de têtes d'attention
            num_iterations: Nombre d'itérations de la dynamique INL
            target_value: Valeur cible initiale
            feedforward_dim: Dimension du feedforward (défaut: 4*d_model)
            dropout: Dropout rate
            use_attention: Si True, utilise attention + INL. Si False, seulement INL (ancien comportement)
        """
        super().__init__()

        self.d_model = d_model
        self.num_iterations = num_iterations
        self.use_attention = use_attention

        if feedforward_dim is None:
            feedforward_dim = 4 * d_model

        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model) if use_attention else None

        # === AJOUT: Multi-Head Attention ===
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True  # Input shape: [batch, seq, dim]
            )
        else:
            # Fallback: projection simple (ancien comportement)
            self.context_projection = nn.Linear(d_model, d_model)

        # INL pour chaque position
        self.inl = IntegratorNeuronLayer(
            hidden_dim=d_model,
            output_dim=d_model,
            target_value=target_value,
            dt=0.1
        )

        # Feedforward (comme dans Transformer)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask (causal mask pour autoregressive)

        Returns:
            output: [batch_size, seq_len, d_model]
            aux_info: Informations auxiliaires de INL
        """
        batch_size, seq_len, d_model = x.shape

        # === ÉTAPE 1: Multi-Head Attention (nouveau) ===
        if self.use_attention:
            # Normalisation pré-attention
            x_norm = self.norm_attn(x)

            # Créer le masque causal pour l'attention autoregressive
            # PyTorch MultiheadAttention attend un masque additive (True = masqué)
            if mask is None:
                # Masque causal: chaque position peut seulement voir les positions précédentes
                attn_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
            else:
                attn_mask = mask

            # Attention multi-tête
            attn_output, attn_weights = self.attention(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=False
            )

            # Connexion résiduelle + dropout
            x = x + self.dropout(attn_output)

            # Le contexte pour INL vient de l'attention
            context = attn_output
        else:
            # Ancien comportement: projection simple
            x_norm = self.norm1(x)
            context = self.context_projection(x_norm)

        # === ÉTAPE 2: Dynamique INL (raffinement) ===
        # Normalisation
        x_norm = self.norm1(x)

        # Initialiser états et vitesses pour tous les tokens
        x_state = x_norm.clone()
        v_state = torch.zeros_like(x_norm)

        # Faire évoluer via dynamique INL (itérations parallèles)
        for iteration in range(self.num_iterations):
            # Reshape pour passer par INL
            # INL attend [batch_size * seq_len, d_model]
            x_flat = x_state.reshape(batch_size * seq_len, d_model)
            v_flat = v_state.reshape(batch_size * seq_len, d_model)
            ctx_flat = context.reshape(batch_size * seq_len, d_model)

            # Forward INL en parallèle pour tous les tokens
            x_next_flat, v_next_flat, aux = self.inl(ctx_flat, x_flat, v_flat)

            # Reshape back
            x_state = x_next_flat.reshape(batch_size, seq_len, d_model)
            v_state = v_next_flat.reshape(batch_size, seq_len, d_model)

        output = x_state
        aux_infos = aux  # Garder les infos de la dernière itération

        # Connexion résiduelle + dropout
        x = x + self.dropout(output)

        # === ÉTAPE 3: Feedforward ===
        x = x + self.ff(self.norm2(x))

        return x, aux_infos


class IntegratorLanguageModel(nn.Module):
    """
    Modèle de langage hybride: Attention + INL.

    Architecture améliorée :
        Token Embedding → Positional Encoding →
        N × (Attention + INL + Feedforward) → LM Head

    Innovations :
    - Multi-head attention pour capturer les dépendances contextuelles
    - Dynamique intégrateur INL pour raffiner les représentations
    - Convergence contrôlée via α, β, g
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_iterations_per_layer: int = 5,
        feedforward_dim: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        tie_weights: bool = True,
        use_attention: bool = True
    ):
        """
        Args:
            vocab_size: Taille du vocabulaire
            d_model: Dimension du modèle
            num_layers: Nombre de couches IntegratorLanguageBlock
            num_heads: Nombre de têtes d'attention (défaut: 8)
            num_iterations_per_layer: Itérations INL par couche
            feedforward_dim: Dimension feedforward
            max_seq_len: Longueur maximale de séquence
            dropout: Dropout rate
            tie_weights: Partager poids embedding et LM head
            use_attention: Si True, architecture hybride Attention+INL. Si False, seulement INL
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Dropout initial
        self.dropout = nn.Dropout(dropout)

        # Stack de couches IntegratorLanguageBlock (hybrides)
        self.layers = nn.ModuleList([
            IntegratorLanguageBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_iterations=num_iterations_per_layer,
                target_value=0.0,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                use_attention=use_attention
            )
            for _ in range(num_layers)
        ])

        # Normalisation finale
        self.final_norm = nn.LayerNorm(d_model)

        # LM head (projection vers vocabulaire)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (partage poids embedding et output)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialisation
        self._init_weights()

    def _init_weights(self):
        """Initialisation des poids."""
        # Embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # LM head
        if not hasattr(self.lm_head.weight, 'data'):
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional [batch_size, seq_len]
            return_aux: Retourner infos auxiliaires INL

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            aux_infos: Optional liste des infos INL par couche
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        x = self.token_embedding(input_ids)  # [B, S, D]

        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Passer par les couches
        all_aux = [] if return_aux else None

        for layer in self.layers:
            x, aux = layer(x, mask=attention_mask)

            if return_aux:
                all_aux.append(aux)

        # Normalisation finale
        x = self.final_norm(x)

        # Projection vers vocabulaire
        logits = self.lm_head(x)  # [B, S, V]

        return logits, all_aux

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Génération autogressive.

        Args:
            input_ids: [batch_size, seq_len] prompt
            max_new_tokens: Nombre de tokens à générer
            temperature: Température de sampling
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Échantillonner ou prendre argmax

        Returns:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward
                logits, _ = self.forward(input_ids)

                # Prendre logits du dernier token
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative prob > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample ou argmax
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self) -> int:
        """Compte nombre de paramètres."""
        return sum(p.numel() for p in self.parameters())


# Fonction helper pour créer un petit modèle de test
def create_small_inl_lm(vocab_size: int = 10000):
    """
    Créer un petit IntegratorLanguageModel pour tests.

    Config ~10M paramètres (comparable à GPT-2 small)
    """
    model = IntegratorLanguageModel(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_iterations_per_layer=3,
        feedforward_dim=1024,
        max_seq_len=512,
        dropout=0.1
    )

    print(f"IntegratorLanguageModel créé:")
    print(f"  Paramètres: {model.get_num_params():,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  d_model: {model.d_model}")
    print(f"  Layers: {model.num_layers}")

    return model


if __name__ == '__main__':
    print("="*60)
    print("IntegratorLanguageModel - Test")
    print("="*60)

    # Créer modèle
    model = create_small_inl_lm(vocab_size=5000)

    # Test forward
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 5000, (batch_size, seq_len))

    print("\nTest forward pass...")
    logits, aux = model(input_ids, return_aux=True)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Aux info layers: {len(aux)}")

    # Test génération
    print("\nTest génération...")
    prompt = torch.randint(0, 5000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, do_sample=True)

    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    print("\n" + "="*60)
    print("✓ IntegratorLanguageModel fonctionne !")
    print("="*60)
