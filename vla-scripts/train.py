import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHeadmulmlpk
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import (
    ProprioProjector,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    PROPRIO_DIM,
    ACTION_TOKEN_IDX,
)
from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.vla.datasets.datasets import  lisaRLDSBatchTransformmul

from experiments.robot.openvla_utils import _load_dataset_stats, find_checkpoint_file, load_component_state_dict

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    pretrained_checkpoint: str = None                # Path to pretrained checkpoint
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 60_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # effvla parameters
    mode: str = None
    action_head_name: str = None
    num_actions_chunk: int = -1
    num_actions_per_token: int = -1
    num_blocks: int = -1
    # fmt: on

def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.pretrained_checkpoint.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        )
        if cfg.use_proprio:
            run_id += "+proprio"
        run_id += "+3rd_img"  # "+3rd_person_img"
        if cfg.num_images_in_input == 2:
            run_id += "+wrist_img"
        run_id += f"+{cfg.num_actions_chunk}act"
        run_id += f"+{cfg.num_actions_per_token}apt"
        run_id += f"+lr{cfg.learning_rate}"
        if cfg.run_id_note is not None:
            run_id += f"+{cfg.run_id_note}"
    return run_id




def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla:OpenVLAForActionPrediction,
    action_head:nn.Module,
    noisy_action_projector:nn.Module,
    proprio_projector:nn.Module,
    batch:dict,
    # action_tokenizer: ActionTokenizer,
    base_tokenizer,
    device_id: str,
    use_l1_regression: bool,
    use_diffusion: bool,
    use_proprio: bool,
    use_film: bool,
    num_patches: int,
    compute_diffusion_l1: bool,
    num_diffusion_steps=None,
    cfg: FinetuneConfig = None,
    current_step: int = 0,
    steps_per_epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        base_tokenizer: Base tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id), # torch.Size([B, seq])
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id), 
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)  # torch.Size([B, seq-1])
    all_actions_mask = batch["input_ids"][:, 1:] ==  ACTION_TOKEN_IDX 
    assert (all_actions_mask.sum(dim=1) == cfg.num_actions_chunk //cfg.num_actions_per_token).all()
    action_token_ids = ground_truth_token_ids[all_actions_mask]  
    assert all([token == base_tokenizer("<ACT>", add_special_tokens=False).input_ids[0] for token in action_token_ids])

    all_actions_mask = torch.cat(
        [
            all_actions_mask,
            torch.zeros((all_actions_mask.shape[0], 1)).bool(),
        ],
        dim=1,
    ) 

    # Get last layer hidden states
    last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
    # Get hidden states for text portion of prompt+response (after the vision patches)
    text_hidden_states = last_hidden_states[:, num_patches:] # [B, seq_len, 4096])   # oft is using -1 to remove the last one, we are more explicit
    batchsize = text_hidden_states.shape[0]
    actions_hidden_states = (
        text_hidden_states[all_actions_mask] 
        .to(torch.bfloat16)
    )   #will auto flatten

    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    actions_hidden_states = actions_hidden_states.reshape(batchsize, cfg.num_actions_chunk//cfg.num_actions_per_token, text_hidden_states.shape[-1])
    if cfg.use_diffusion:
        pass
    elif cfg.use_l1_regression:
        predicted_actions = action_head.module.predict_action(actions_hidden_states)
        action_loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions) 
        ground_truth_curr_action = ground_truth_actions[:, 0]
        predicted_curr_action = predicted_actions[:, 0]
        ground_truth_next_actions = ground_truth_actions[:, 1:]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
        next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
    else:
        raise ValueError(f"Invalid: cfg.use_diffusion {cfg.use_diffusion} ,  cfg.use_l1_regression {cfg.use_l1_regression}")
    
    
    tokenloss = output.loss
    loss = tokenloss*0.01 + action_loss*0.99
    metrics.update(
        {
            "loss_value": loss.item(),  # Detached value for logging
            "tokenloss": tokenloss.item(),
            "action_loss": action_loss.item(),
        }
    )
    if cfg.use_l1_regression:
        metrics.update(
            {
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),

            }
        )
    
    return loss, metrics

def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg: FinetuneConfig,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")


    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    
    # need to save config.json
    if distributed_state.is_main_process and cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        config_path = checkpoint_dir / "config.json"
        base_config = base_vla.config.to_dict()
        base_config["norm_stats"] = train_dataset.dataset_statistics # adhoc solution 
        with open(config_path, "w") as f:
            json.dump(base_config, f, indent=2) 

    dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    # action_tokenizer,
    base_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                # action_tokenizer=action_tokenizer,
                base_tokenizer=base_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                cfg=cfg,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    print("cfg", cfg)
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    
    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    world_size = torch.distributed.get_world_size()
    run_id += f"_{world_size}gpus"

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {cfg.num_actions_chunk}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if cfg.resume and cfg.pretrained_checkpoint:
        # Resume training from a pretrained checkpoint
        if distributed_state.is_main_process:
            print(f"Resuming training from checkpoint: {cfg.pretrained_checkpoint}")
        
        if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
            # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            
            # Update config.json and sync model files
            if distributed_state.is_main_process:
                update_auto_map(cfg.pretrained_checkpoint)
                check_model_logic_mismatch(cfg.pretrained_checkpoint)
        
        # Wait for model files to be synced
        dist.barrier()
        
        # Load the base model first
        base_vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.vla_path, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        base_vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        llm_dim = base_vla.llm_dim

        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        num_added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<ACT>"]})
        assert num_added_tokens == 1
        assert processor.tokenizer.convert_tokens_to_ids('<ACT>') == 32001
    
        vla_lora = PeftModel.from_pretrained(
            base_vla, 
            model_id=cfg.pretrained_checkpoint, 
            subfolder="lora_adapter",
            is_trainable=True
        )
        # vla_lora = vla_lora.merge_and_unload() # training so we don't merge, eval needs merge to load  model.config.norm_stats
        
        dist.barrier()

        vla_lora = vla_lora.to(device_id)
        vla_lora.train()

        if distributed_state.is_main_process:
            _load_dataset_stats(vla_lora, cfg.pretrained_checkpoint)
        vla_lora.forward = vla_lora.lisa_forward
        vla_lora.predict_action = vla_lora.lisa_predict_action

        vla = wrap_ddp(vla_lora, device_id, find_unused=True)

        if cfg.action_head_name == "mlp":
            action_head = L1RegressionActionHeadmulmlpk(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_actions_chunk=cfg.num_actions_chunk, num_actions_per_token=cfg.num_actions_per_token, num_blocks=cfg.num_blocks)
            
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)
        action_head = action_head.to(torch.bfloat16)
        action_head = action_head.to(device_id)
        action_head.train()
        action_head = wrap_ddp(action_head, device_id, find_unused=True)

    else:
        if cfg.pretrained_checkpoint != None:
            raise ValueError("pretrained_checkpoint must pass resume args")
        if cfg.resume:
            raise ValueError("resume must pass pretrained_checkpoint args")
        
        if model_is_on_hf_hub(cfg.vla_path):
            # Download model directly from Hugging Face Hub
            vla_download_path = snapshot_download(repo_id=cfg.vla_path)
            # Overwrite VLA path
            cfg.vla_path = vla_download_path
        else:
            # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        if distributed_state.is_main_process:
            update_auto_map(cfg.vla_path)
            check_model_logic_mismatch(cfg.vla_path)

        # Wait for model files to be synced
        dist.barrier()

        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla : OpenVLAForActionPrediction = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device_id)
        
        vla.forward = vla.lisa_forward
        vla.predict_action = vla.lisa_predict_action

        num_added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<ACT>"]})

        assert num_added_tokens == 1
        assert processor.tokenizer.convert_tokens_to_ids('<ACT>') == 32001

        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

        # LoRA setup
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=16, 
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

        vla = wrap_ddp(vla, device_id, find_unused=True)

        if cfg.use_proprio:
            proprio_projector = init_module(
                ProprioProjector,
                "proprio_projector",
                cfg,
                device_id,
                {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            )

        if cfg.action_head_name == "mlp":
            action_head = init_module(
                L1RegressionActionHeadmulmlpk,
                "action_head",
                cfg,
                device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM, "num_actions_chunk": cfg.num_actions_chunk, "num_actions_per_token": cfg.num_actions_per_token, "num_blocks": cfg.num_blocks},
            to_bf16=True,
            )


    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1 # information mapping to one token

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input == 2

    # Create training and optional validation datasets
    batch_transform = lisaRLDSBatchTransformmul(
        action_tokenizer=None,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        token_num= cfg.num_actions_chunk //cfg.num_actions_per_token,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        cfg=cfg,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
            cfg=cfg,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "tokenloss": deque(maxlen=cfg.grad_accumulation_steps),
        "action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "grad_norm": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            total_samples = len(train_dataset)
            num_gpus = torch.cuda.device_count()
            steps_per_epoch = total_samples // (cfg.batch_size * cfg.grad_accumulation_steps * num_gpus)
            
            # Compute gradient step index and log step
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            
            if distributed_state.is_main_process and batch_idx == 0:
                print(f"Total samples: {total_samples}")
                print(f"Number of GPUs: {num_gpus}")
                print(f"Steps per epoch: {steps_per_epoch}")

            
            # compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                # noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                noisy_action_projector=None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                # action_tokenizer=action_tokenizer,
                base_tokenizer=processor.tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                # compute_diffusion_l1=compute_diffusion_l1,
                compute_diffusion_l1=False,
                num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                cfg=cfg,
                current_step=log_step,
                steps_per_epoch=steps_per_epoch,
            )

            normalized_loss = loss / cfg.grad_accumulation_steps

            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            # gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10)# mlp16,lr1e-4最大就是一开始10.

            if distributed_state.is_main_process:
                grad_norms = [torch.norm(p.grad) for p in trainable_params if p.grad is not None]
                if grad_norms:
                    total_norm = torch.norm(torch.stack(grad_norms), 2)
                else:
                    total_norm = 0
                recent_metrics["grad_norm"].append(total_norm.item())

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            # log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
            #     # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/lr": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0 and log_step > 6*cfg.save_freq:
                print(f"Saving checkpoint, smoothened_metrics on checkpoint {log_step} is")
                print(smoothened_metrics)
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    # noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    noisy_action_projector=None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                ) 

            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    # noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    # action_tokenizer=action_tokenizer,
                    base_tokenizer=processor.tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()
                exit()

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
