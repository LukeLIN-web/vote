"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math
from diffusers import DDIMScheduler
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from transformers import ViTModel, ViTConfig

# original oft head
class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            # num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
            num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            # nn.ReLU(),
            nn.SiLU(), # 视觉任务表现较好. 
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        self.relu = nn.SiLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

class L1RegressionActionHeadmulmlpk(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_actions_chunk=-999,
        num_actions_per_token=-999,
        num_blocks=16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_per_token = num_actions_per_token
        self.num_actions_chunk = num_actions_chunk
        if num_actions_chunk < 0 or num_actions_per_token <0 or num_blocks <0:
            raise ValueError("num_actions_chunk, num_actions_per_token, num_blocks must be set")
        self.model = MLPResNet(
            num_blocks=num_blocks, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim*num_actions_per_token
        )

    def predict_action(self, actions_hidden_states:torch.Tensor):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, num of tokens, hidden_dim)  
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]     
        action = self.model(actions_hidden_states) # output shape: (batch_size, num of tokens, action_dim*num_actions_per_token)
        action = action.reshape(batch_size, self.num_actions_chunk, self.action_dim)
        return action 

class MLPResNetBlockV2(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x_ffn = self.ffn(x)
        x_dropped = self.dropout(x_ffn)
        x = x_dropped + identity
        return x

class L1RegressionActionHeadFunnel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_actions_chunk: int,
                 num_actions_per_token: int,
                 num_blocks: int,
                 expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.num_actions_per_token = num_actions_per_token

        # 1. 输入投射层
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        # 2. ResNet 主体
        self.resnet_body = nn.ModuleList()
        for _ in range(num_blocks):
            self.resnet_body.append(
                MLPResNetBlockV2(dim=hidden_dim, expansion=expansion, dropout=dropout)
            )

        # 3. 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim*num_actions_per_token)
        )

    def predict_action(self, actions_hidden_states: torch.Tensor):
        """
        Predict actions using funnel MLP design.
        
        Args:
            actions_hidden_states: (batch_size, num_tokens, input_dim)
        Returns:
            action: (batch_size, num_actions_chunk, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        
        # action = self.model(actions_hidden_states)  # (batch_size, num_tokens, action_dim*num_actions_per_token)
        
        # Apply input projection
        x = self.input_proj(actions_hidden_states)
        
        # Apply ResNet blocks
        for block in self.resnet_body:
            x = block(x)
        
        # Apply output head
        action = self.output_head(x)  # (batch_size, num_tokens, action_dim*num_actions_per_token)
        
        # Reshape to final output
        action = action.reshape(batch_size, self.num_actions_chunk, self.action_dim)
        return action


class ViTActionHead(nn.Module):
    def __init__(self, 
                 pretrained_model_name: str = "timm/deit3_large_patch16_224.fb_in22k_ft_in1k", # base只有 12层, 太小了
                 input_dim: int = 4096,             # LLM 输出维度
                 vit_hidden_dim: int = 768,         
                 action_dim: int = 7,
                 num_actions_chunk: int = 16,
                 num_actions_per_token: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.vit_hidden_dim = vit_hidden_dim
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.num_actions_per_token = num_actions_per_token
        
        self.pre_proj = nn.Linear(input_dim, vit_hidden_dim)# Linear projection: LLM 4096 → ViT 768

        vit_config = ViTConfig.from_pretrained(pretrained_model_name)
        vit_config.hidden_size = vit_hidden_dim
        vit_config.num_channels = 1
        vit_config.image_size = 1
        vit_config.patch_size = 1

        self.encoder = ViTModel(vit_config).encoder
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_actions_chunk, vit_hidden_dim))# 位置编码（可训练）
        self.action_proj = nn.Linear(vit_hidden_dim, action_dim*num_actions_per_token)# 输出 head: 768 → action_dim
        

    def predict_action(self, x:torch.Tensor):
        if x.ndim == 2:
            B, D = x.shape
            num_tokens = 1
            x = x.unsqueeze(1)# [B, 1, D]#x中间插入一个维度
        else:
            B, num_tokens, hidden_dim = x.shape # prepare for future multiple tokens' hidden states
            assert hidden_dim == self.input_dim, f"Expected D={self.input_dim}, got {hidden_dim}"
        
        x = self.pre_proj(x)                  # [B, T, vit_hidden_dim]
        x = x + self.pos_embedding[:, :num_tokens, :]  # 加位置编码
        x = self.encoder(x)[0]                # [B, T, vit_hidden_dim]
        x = self.action_proj(x)               # [B, T, action_dim]
        action = x.reshape(B, self.num_actions_chunk, self.action_dim)# 训练的时候就是16个一起, 所以不能和训练的chunk token不一样.  
        return action



## diffusion 用到的. 
class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb
class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps=100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
        self.num_diffusion_steps = num_diffusion_steps
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred
