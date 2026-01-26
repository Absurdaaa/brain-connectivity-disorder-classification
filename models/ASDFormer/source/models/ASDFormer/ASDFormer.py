import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from .activations import ACT2FN
from omegaconf import DictConfig
from ..base import BaseModel
import pickle
import torch.nn.functional as F

class AdvancedMLP(nn.Module):
    def __init__(self, config, input_dim, output_dim, hidden_dim=128, layers=2, dropout_prob=0.0):
        super().__init__()
        self.layers = layers
        if dropout_prob == 0.0:
            self.dropout = nn.Dropout(config.model.dropout)
        else:
            self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

        if self.layers == 1:
            # Single linear layer: input → output
            self.fc = nn.Linear(input_dim, output_dim)
        elif self.layers == 2:
            # Two-layer MLP: input → hidden → output
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        else:
            # Raise error for unsupported number of layers
            raise ValueError(f"Unsupported number of layers: {self.layers}. Only 1 or 2 are allowed.")

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.layers == 1:
            x = self.fc(x)
        elif self.layers == 2:
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            # Just in case forward is somehow called with bad state
            raise RuntimeError(f"Forward called with invalid layers setting: {self.layers}")
        return x




class ASDFormerNodeFeature(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Number of Regions of Interest (ROIs) in the brain network graph
        self.num_ROIs = config.model.num_ROIs
        
        # Dimensionality of the feature embeddings per ROI
        self.embedding_dim = config.model.embedding_dim
        
        # Hidden dimension for the MLP encoder
        self.feature_encoder_hidden = config.model.feature_encoder_hidden
        
        # Number of layers in the MLP encoder
        self.feature_encoder_layer = config.model.feature_encoder_layer

        # ------------------------------------------------------------------
        # Shared feature encoder across all ROIs:
        # - AdvancedMLP: encodes node-level features (FC, etc.)
        # - LayerNorm: ensures normalized embeddings for stability
        # ------------------------------------------------------------------
        self.shared_mlp = AdvancedMLP(
            config, 
            self.num_ROIs, 
            self.embedding_dim, 
            self.feature_encoder_hidden, 
            self.feature_encoder_layer
        )
        self.shared_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        FC_matrix: torch.Tensor,     # Functional Connectivity Matrix, shape (B, N, D)
    ) -> torch.Tensor:

        # Extract dimensions: 
        # B = batch size, N = number of ROIs (nodes), D = feature dimension
        B, N, D = FC_matrix.shape

        # Pass functional connectivity features through shared MLP
        # Output shape: (B, N, embedding_dim)
        x = self.shared_mlp(FC_matrix)     

        # Apply layer normalization for feature stabilization
        # Keeps embeddings well-scaled across batches
        x = self.shared_norm(x) 

        # Return ROI-level feature embeddings for downstream modules
        return x




class ASDFormerMultiheadAttention(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        # Core embedding dimensions
        self.embedding_dim = config.model.embedding_dim
        self.kdim = config.model.embedding_dim  # Key dimension
        self.vdim = config.model.embedding_dim  # Value dimension

        # Whether query, key, and value projections all have same dimensions
        self.qkv_same_dim = self.kdim == config.model.embedding_dim and self.vdim == config.model.embedding_dim

        # Multi-head attention settings
        self.num_heads = config.model.num_attention_heads
        self.attention_dropout_module = torch.nn.Dropout(p=config.model.dropout, inplace=False)

        # Dimension of each attention head
        self.head_dim = config.model.embedding_dim // config.model.num_attention_heads
        if not (self.head_dim * config.model.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        
        # Scaling factor for query projection (standard in attention)
        self.scaling = self.head_dim**-0.5

        # Linear projections for query, key, value
        self.k_proj = nn.Linear(self.kdim, config.model.embedding_dim, bias=config.model.bias)
        self.v_proj = nn.Linear(self.vdim, config.model.embedding_dim, bias=config.model.bias)
        self.q_proj = nn.Linear(config.model.embedding_dim, config.model.embedding_dim, bias=config.model.bias)

        # Output projection to recombine head outputs
        self.out_proj = nn.Linear(config.model.embedding_dim, config.model.embedding_dim, bias=config.model.bias)
        
        # Initialize weights
        self.reset_parameters()


    def reset_parameters(self):
        """Custom initialization for Q, K, V, and output projection weights."""

        if self.qkv_same_dim:
            # Use scaled Xavier init for faster convergence when q, k, v share dim
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        # Standard Xavier init for output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.LongTensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  
        """
        Compute scaled dot-product multi-head attention.
        Returns:
            - attn: attention output (T, B, E)
            - attn_weights: averaged attention weights (B, T, S)
        """
        
        # During inference we want weights, but not per-head weights
        if self.training == False:
            need_head_weights = False
            need_weights = True
        else:
            need_head_weights = False
            need_weights = True

        # Query shape check
        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len  # default to self-attention case

        if not (embedding_dim == self.embedding_dim):
            raise AssertionError(
                f"The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim"
                f" {self.embedding_dim}."
            )
        if not (list(query.size()) == [tgt_len, bsz, embedding_dim]):
            raise AssertionError("Query size incorrect in ASDFormer, compared to model dimensions.")

        # Validate key/value batch sizes if provided
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if (key_bsz != bsz) or (value is None) or not (src_len, bsz == value.shape[:2]):
                    raise AssertionError(
                        "The batch shape does not match the key or value shapes provided to the attention."
                    )

        # Linear projections for Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        # Scale queries (stabilizes gradients for large dot-products)
        q *= self.scaling

        # Reshape into heads: (num_heads*B, seq_len, head_dim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Validate key shape
        if (k is None) or not (k.size(1) == src_len):
            raise AssertionError("The shape of the key generated in the attention is incorrect")

       
        # Compute raw attention scores: (num_heads*B, T, S)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError("The attention weights generated do not match the expected dimensions.")


        # Softmax normalization over source length (last dim)
        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)

        # Apply dropout to attention probabilities
        attn_probs = self.attention_dropout_module(attn_weights)

        # Weighted sum of values
        if v is None:
            raise AssertionError("No value generated")
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError("The attention generated do not match the expected dimensions.")

        # Recombine heads: (T, B, E)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn: torch.Tensor = self.out_proj(attn)

        # Aggregate attention weights
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights



class ASDFormerEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        # Dropout applied to input node features for regularization
        self.dropout_module = torch.nn.Dropout(p=config.model.dropout, inplace=False)

        # Node feature extractor (e.g., from FC_matrix → embeddings)
        self.node_feature = ASDFormerNodeFeature(config)

        # Stack of Transformer-style encoder layers
        self.layers = nn.ModuleList([])
        self.layers.extend([
            ASDFormerEncoderLayer(config) for _ in range(config.model.num_hidden_layers)
        ])

    def forward(
        self,
        FC_matrix: torch.LongTensor,  # Functional Connectivity input (B, N, D)
    ) -> Tuple[
        torch.Tensor,                  # Final node embeddings (T, B, E)
        List[torch.Tensor]             # List of attention maps from each layer
    ]:
        """
        Forward pass of the ASDFormer encoder.

        Args:
            FC_matrix: Functional connectivity features for all nodes.
                       Shape: (B, N, D) where:
                           B = batch size
                           N = number of nodes (ROIs)
                           D = feature dimension (number of nodes)

        Returns:
            - input_nodes: final encoded node embeddings (T, B, E)
            - all_attn_softmax: list of attention weights from each encoder layer
        """

        # Encode raw ROI features → embeddings
        input_nodes = self.node_feature(FC_matrix)

        # Apply dropout to embeddings
        input_nodes = self.dropout_module(input_nodes)

        # Transformer convention: transpose to (T, B, E)
        # T = number of tokens (nodes), B = batch size, E = embedding dim
        input_nodes = input_nodes.transpose(0, 1)

        # Track attention softmax distributions from all layers
        all_attn_softmax = []

        # Pass through each encoder layer
        for layer in self.layers:
            input_nodes, attn = layer(input_nodes)

            # Store per-layer attention for analysis/visualization
            all_attn_softmax.append(attn)

        # Return final embeddings + per-layer attentions
        return input_nodes, all_attn_softmax


class ASDFormerEncoderLayer(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        # Embedding dimension for all node representations
        self.embedding_dim = config.model.embedding_dim

        # Dropouts: applied after attention and FFN for regularization
        self.dropout_module = torch.nn.Dropout(p=config.model.dropout, inplace=False)
        self.activation_dropout_module = torch.nn.Dropout(p=config.model.dropout, inplace=False)

        # Nonlinear activation (e.g., GELU, ReLU) used in the feed-forward network
        self.activation_fn = ACT2FN[config.model.activation_fn]

        # Multi-head self-attention module
        self.self_attn = ASDFormerMultiheadAttention(config)

        # Layer normalization after attention block
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Feed-forward network (two-layer MLP with activation in between)
        self.fc1 = nn.Linear(self.embedding_dim, config.model.ffn_embedding_dim)  # expand hidden size
        self.fc2 = nn.Linear(config.model.ffn_embedding_dim, self.embedding_dim)  # project back to embedding dim

        # Layer normalization after FFN block
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        # Initialize FC weights
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


    def forward(
        self,
        input_nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of one Transformer encoder layer.
        Implements: LayerNorm → MultiheadAttention → Residual → FFN → Residual
        """

        # -------- Self-Attention Block --------
        residual = input_nodes
        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,

        )
        input_nodes = self.dropout_module(input_nodes)   # apply dropout after attention
        input_nodes = residual + input_nodes             # residual connection
        input_nodes = self.self_attn_layer_norm(input_nodes)  # normalize

        # -------- Feed-Forward Block --------
        residual = input_nodes
        input_nodes = self.activation_fn(self.fc1(input_nodes))   # expand & activate
        input_nodes = self.activation_dropout_module(input_nodes) # dropout on hidden activations
        input_nodes = self.fc2(input_nodes)                       # project back down
        input_nodes = self.dropout_module(input_nodes)            # dropout again
        input_nodes = residual + input_nodes                      # residual connection
        input_nodes = self.final_layer_norm(input_nodes)          # normalize

        # Return updated node embeddings + attention weights
        return input_nodes, attn


class Expert(nn.Module):
    def __init__(self, config, pooling_type, input_dim):
        super().__init__()
        # Input embedding dimension (F in forward pass)
        self.embedding_dim = input_dim

        # Pooling strategy: "topk1" or "topk2"
        self.pooling_type = pooling_type

        # Hyperparameters for top-k pooling
        self.k1 = config.model.k1
        self.k2 = config.model.k2

        # Classification task settings
        self.num_classes = config.model.num_classes

        # Hidden dimensions and number of layers for attention & classifier
        self.expert_hidden = config.model.expert_hidden
        self.expert_layer = config.model.expert_layer
        self.classifier_hidden = config.model.classifier_hidden
        self.classifier_layer = config.model.classifier_layer


        # Attention network for top-k1 pooling
        if self.pooling_type == "topk1":
            self.attention1 = AdvancedMLP(
                config, self.embedding_dim, 1, self.expert_hidden, self.expert_layer
            )

        # Attention network for top-k2 pooling
        if self.pooling_type == "topk2":
            self.attention2 = AdvancedMLP(
                config, self.embedding_dim, 1, self.expert_hidden, self.expert_layer
            )

        # Final classifier (maps pooled representation → class logits)
        self.classifier = AdvancedMLP(
            config,
            self.embedding_dim,
            self.num_classes,
            self.classifier_hidden,
            self.classifier_layer
        )
    def forward(self, x):  # x: (B, N_cluster, F)
        """
        Forward pass for one expert.

        Args:
            x: Node features for clustered ROIs.
               Shape = (B, N_cluster, F)
                 B = batch size
                 N_cluster = number of clusters/nodes
                 F = embedding_dim

        Returns:
            - class_logits: tensor of shape (B, num_classes)
            - topk_info: tensor of shape (B, k, 2) with [index, attention weight]
        """

        if self.pooling_type == "topk1":
            
            # Step 1: Compute attention logits
            attention_logit = self.attention1(x).squeeze(-1)  # (B, N)

            # Step 2: Select top-k per sample
            topk_values, topk_indices = torch.topk(attention_logit, self.k1, dim=-1)  # (B, k)

            # Step 3: Create a mask for top-k indices
            mask = torch.zeros_like(attention_logit)  # (B, N)
            mask.scatter_(dim=1, index=topk_indices, value=1.0)  # 1 where top-k, 0 elsewhere

            # Step 4: Apply softmax only over top-k elements
            masked_logits = attention_logit.masked_fill(mask == 0, float('-inf'))  # Exclude non-topk
            topk_attention_weights = F.softmax(masked_logits, dim=-1)  # Re-normalize only top-k

            # Step 5: Get attention weights at top-k indices
            topk_attention = topk_attention_weights.gather(1, topk_indices)  # (B, k)

            # Step 5: Apply attention weights
            weighted_x = x * topk_attention_weights.unsqueeze(-1)  # (B, N, F)

            # Step 6: Aggregate features
            pooled = weighted_x.sum(dim=1)  # (B, F)

        elif self.pooling_type == "topk2":
            
            # Step 1: Compute attention logits
            attention_logit = self.attention2(x).squeeze(-1)  # (B, N)

            # Step 2: Select top-k per sample
            topk_values, topk_indices = torch.topk(attention_logit, self.k2, dim=-1)  # (B, k)

            # Step 3: Create a mask for top-k indices
            mask = torch.zeros_like(attention_logit)  # (B, N)
            mask.scatter_(dim=1, index=topk_indices, value=1.0)  # 1 where top-k, 0 elsewhere

            # Step 4: Apply softmax only over top-k elements
            masked_logits = attention_logit.masked_fill(mask == 0, float('-inf'))  # Exclude non-topk
            topk_attention_weights = F.softmax(masked_logits, dim=-1)  # Re-normalize only top-k

            # Step 5: Get attention weights at top-k indices
            topk_attention = topk_attention_weights.gather(1, topk_indices)  # (B, k)

            # Step 5: Apply attention weights
            weighted_x = x * topk_attention_weights.unsqueeze(-1)  # (B, N, F)

            # Step 6: Aggregate features
            pooled = weighted_x.sum(dim=1)  # (B, F)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        topk_info = torch.stack([topk_indices, topk_attention], dim=-1)  # (B, k, 2)

        return self.classifier(pooled), topk_info  # each row: (index, attention)


class MixtureofExperts(nn.Module):
    def __init__(self, config: DictConfig, forward_dim):
        super().__init__()
   
        # Number of ROIs (nodes/clusters in graph)
        self.num_ROIs = config.model.num_ROIs

        # Define the available experts based on pooling strategies
        self.pooling_types = ['topk1', 'topk2']
        self.n_experts = len(self.pooling_types)

        # Hyperparameters for gate network
        self.gate_hidden = config.model.gate_hidden
        self.gate_layer = config.model.gate_layer

        # Initialize expert networks (one per pooling type)
        # Each expert outputs class logits + topk_info
        self.experts = nn.ModuleDict({
            pooling: Expert(config, pooling, forward_dim)
            for pooling in self.pooling_types
        })

        # Input to gating network is the flattened representation of all nodes
        # Shape: (B, num_ROIs * forward_dim)
        gate_input_dim = self.num_ROIs * forward_dim

        # Gating MLP: maps input features → expert weight logits
        self.gate = AdvancedMLP(
            config,
            gate_input_dim,
            self.n_experts,   # one output weight per expert
            self.gate_hidden,
            self.gate_layer
        )

    def forward(self, x):
        B, N, D = x.size()
        """
        Forward pass for Mixture-of-Experts.

        Args:
            x: Input node embeddings
               Shape = (B, N, D)
                 B = batch size
                 N = number of ROIs
                 D = embedding dimension

        Returns:
            - X_out: Weighted expert prediction logits, shape (B, num_classes)
            - gate_weights: Softmax weights per expert, shape (B, n_experts, 1)
            - combined_topk_info: Top-k indices + attention scores from all experts, shape (B, total_k, 2)
        """

        # -------------------------------
        # Step 1: Gating network
        # -------------------------------
        gate_input = x.view(B, -1)                  # [n, num_ROIs * d]
        gate_out = self.gate(gate_input)                  # [n, num_experts]
        gate_weights = torch.softmax(gate_out, dim=1)     # [n, num_experts]

        # -------------------------------
        # Step 2: Expert forward passes
        # -------------------------------
        expert_outputs = []
        all_topk_infos = []

        for pooling_type in self.pooling_types:
            # Each expert processes the same input `x` with different pooling logic
            out, topk_info = self.experts[pooling_type](x) 
            expert_outputs.append(out.unsqueeze(1))     # Add expert dim -> (B, 1, num_classes)
            all_topk_infos.append(topk_info)          # (B, k, 2)

        # Stack expert predictions -> (B, n_experts, num_classes)
        expert_outputs = torch.cat(expert_outputs, dim=1)

        # Combine top-k details from all experts -> (B, total_k, 2)
        combined_topk_info = torch.cat(all_topk_infos, dim=1)

        # -------------------------------
        # Step 3: Weighted expert aggregation
        # -------------------------------
        gate_weights = gate_weights.unsqueeze(2)          # [n, num_experts, 1]
        X_out = torch.sum(gate_weights * expert_outputs, dim=1)  # [n, 2]

        return X_out, gate_weights, combined_topk_info


class ASDFormerModel(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        # Embedding dimension for all nodes
        self.embedding_dim = config.model.embedding_dim

        # Dimensionality reduction ratio for embeddings before decoding
        self.dim_reduc_ratio = config.model.dim_reduc_ratio

        # Core Transformer-based encoder (stack of ASDFormerEncoderLayers)
        self.encoder = ASDFormerEncoder(config)

        # Parameters for optional dimensionality reduction MLP
        dim_reduc_hidden = config.model.dim_reduc_hidden
        dim_reduc_layer = config.model.dim_reduc_layer


        # -------------------------------------------------------------
        # Optional dimensionality reduction stage:
        # If dim_reduc_ratio != 1, embeddings are projected to a smaller space
        # via an MLP + LayerNorm before being passed to the decoder.
        # -------------------------------------------------------------
        if config.model.dim_reduc_ratio != 1:
            self.reduced_embedding = self.embedding_dim // config.model.dim_reduc_ratio
            self.dim_reduction = AdvancedMLP(config, self.embedding_dim, self.reduced_embedding, dim_reduc_hidden, dim_reduc_layer)
            self.dim_reduction_norm = nn.LayerNorm(self.reduced_embedding)

        else:
            self.reduced_embedding = self.embedding_dim

        # Final decoder: Mixture of Experts (aggregates multiple specialized "experts")
        self.decoder = MixtureofExperts(config, self.reduced_embedding)


    def forward(
        self,
        FC_matrix: torch.LongTensor,   # Functional connectivity matrix: (B, N, N)
    ) -> Tuple[
        torch.LongTensor,              # Head outputs (final predictions/representations)
        List[torch.Tensor],            # Attention maps from encoder layers
        List[torch.Tensor],            # Gating weights from MoE
        List[torch.Tensor]             # Top-k expert selection info
    ]:
        """
        Forward pass of ASDFormerModel:
        1. Encode node features via Transformer encoder.
        2. (Optional) Apply dimensionality reduction to embeddings.
        3. Decode reduced embeddings via Mixture of Experts.
        """

        # ---- Encoder ----
        # inner_states: (T, B, E) = contextualized node embeddings
        # all_attn_softmax: list of per-layer attention maps
        inner_states, all_attn_softmax = self.encoder( FC_matrix )

        # Switch back to (B, N, E) format for downstream modules
        input_nodes = inner_states.transpose(0, 1).contiguous()

        # ---- Dimensionality Reduction (optional) ----
        if self.dim_reduc_ratio != 1:
            reduced_out = self.dim_reduction(input_nodes)
            reduced_out = self.dim_reduction_norm(reduced_out)
        else:
            reduced_out = input_nodes

        # ---- Decoder (Mixture of Experts) ----
        # head_outputs: final predictions per expert
        # gate_weights: learned gating scores for experts
        # topk_info: top-k expert selections per input
        head_outputs, gate_weights, topk_info = self.decoder(reduced_out)

        # Return outputs + intermediate diagnostics
        return head_outputs, all_attn_softmax, gate_weights, topk_info


class ASDFormer(BaseModel):

    def __init__(self, config: DictConfig):
        super().__init__()

        # Core encoder model (stack of ASDFormer encoder layers)
        self.encoder = ASDFormerModel(config)

        # Load precomputed node-to-cluster mapping
        # Example: {0: 1, 1: 3, ..., 199: 7}, meaning each ROI (node) belongs to some cluster
        with open('node_clus_map.pickle', 'rb') as handle:
            self.node_clus_map = pickle.load(handle)

        # Internal buffers to store intermediate results for inspection
        self.all_attn_softmax = None   # attention distributions from all encoder layers
        self.gates = None              # gating values (if used in encoder)
        self.topk_info = None          # top-k selection info (if used in encoder)
        
    def rearrange_node_feature(self, node_feature, rearranged_indices):
        """
        Rearranges node-level features according to cluster mapping.
        Args:
            node_feature: tensor of shape (B, N, N) or (B, N, D)
            rearranged_indices: order of indices from node_clus_map.keys()
        Returns:
            node_feature_rearranged: reordered tensor along both node axes
        """
        node_feature_rearranged = node_feature[:, rearranged_indices, :]
        node_feature_rearranged = node_feature_rearranged[:, :, rearranged_indices]
        return node_feature_rearranged
    
    def rearrange_time_series(self, time_series, rearranged_indices):
        """
        Rearrange the time_series (which has a temporal dimension) according to node_clus_map.
        Args:
            time_series: tensor of shape (B, N, T)
            rearranged_indices: node ordering from node_clus_map.keys()
        Returns:
            time_series_rearranged: tensor of shape (B, N, T) but with reordered nodes
        """
        time_series_rearranged = time_series[:, rearranged_indices, :]
        return time_series_rearranged


    # -----------------------------
    # Accessors for stored outputs
    # -----------------------------
    def get_all_attn_softmax(self) -> List[torch.Tensor]:
        """Retrieve attention maps from all layers."""
        return self.all_attn_softmax

    def get_gates(self) -> List[torch.Tensor]:
        """Retrieve stored gating values (if computed by encoder)."""
        return self.gates
    
    def get_topk_info(self) -> List[torch.Tensor]:
        """Retrieve stored top-k selection information (if computed by encoder)."""
        return self.topk_info
    
    
    # -----------------------------
    # Forward Pass
    # -----------------------------
    def forward(
        self,
        time_series: torch.LongTensor, #[B, nodes, timesteps]
        FC_matrix: torch.LongTensor, #[B, nodes, nodes]
        PC_matrix: torch.LongTensor, #[B, nodes, nodes]
    ) -> torch.tensor:
        
        """
        Forward pass through ASDFormer:
        1. Rearrange inputs according to node cluster mapping.
        2. Encode rearranged node features with ASDFormer encoder.
        3. Store auxiliary outputs (attention, gates, top-k).
        """

        # Rearrange inputs to align with cluster ordering
        time_series_rearranged = self.rearrange_time_series(time_series, list(self.node_clus_map.keys()))
        FC_matrix_rearranged = self.rearrange_node_feature(FC_matrix, list(self.node_clus_map.keys()))
        PC_matrix_rearranged = self.rearrange_node_feature(PC_matrix, list(self.node_clus_map.keys()))

        # Pass rearranged connectivity matrices into encoder
        encoder_outputs, all_attn_softmax, gates, topk_info = self.encoder(
            FC_matrix_rearranged,
        )

        # Store encoder-side attention and auxiliary outputs for analysis
        self.all_attn_softmax = all_attn_softmax
        self.gates = gates
        self.topk_info = topk_info

        # Return final node embeddings
        return encoder_outputs
