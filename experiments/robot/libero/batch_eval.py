import os
import subprocess
import argparse
import sys
from pathlib import Path
from multiprocessing import Process
from datetime import datetime
import time

predefined_ckpts = [
    "juyil/llama3.2-1B-spatial",
]

def run_libero_eval(ckpt_path, task_suite_name, gpu_id, log_base_dir="eval_logs"):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    name = Path(ckpt_path).name
    log_dir = f"{log_base_dir}/{name}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"🚀 Starting evaluation for checkpoint directory: {ckpt_path}")
    print(f"📊 Using GPU: {gpu_id}")
    print(f"📝 Log directory: {log_dir}")
    print("=" * 70)
    
    cmd = [
        sys.executable, "run_libero_eval.py",
        "--model_family", "openvla",
        "--base_vla_path", "juyil/prismatic-llama3.2-dinosiglip-224px-1b-vlm",
        "--pretrained_checkpoint", str(ckpt_path),
        "--task_suite_name", task_suite_name,
        "--center_crop", "True",
        "--use_proprio", "False",
        "--num_images_in_input", "1",
        "--use_l1_regression", "True",
        "--use_diffusion", "False",
        "--use_film", "False",
        "--num_actions_chunk", "8",
        "--num_actions_per_token", "8",
        "--num_blocks", "4",
        "--hidden_dim", "2048",
        "--mode", "mul",
        "--action_head_name", "fel"
    ]
    
    log_file = os.path.join(log_dir, f"{name}.log")
    err_file = os.path.join(log_dir, f"{name}.err")
    
    try:
        with open(log_file, "w") as fout, open(err_file, "w") as ferr:
            result = subprocess.run(cmd, stdout=fout, stderr=ferr, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ Successfully completed evaluation for: {ckpt_path}")
            print(f"📄 Log saved to: {log_file}")
        else:
            print(f"❌ Failed evaluation for: {ckpt_path}")
            print(f"📄 Log saved to: {log_file}")
            print(f"❌ Error log saved to: {err_file}")
            
            try:
                with open(err_file, "r") as f:
                    error_lines = f.readlines()
                    if error_lines:
                        print("Last few error lines:")
                        for line in error_lines[-5:]:
                            print(f"  {line.strip()}")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Exception during evaluation for {ckpt_path}: {e}")
        try:
            with open(err_file, "a") as ferr:
                ferr.write(f"Exception: {e}\n")
        except:
            pass

def get_checkpoint_dirs(parent_dir):
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        print(f"❌ Error: Parent directory not found: {parent_dir}")
        return []
    
    subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"⚠️ Warning: No subdirectories found in {parent_dir}")
        return []
    
    return sorted(subdirs)

def launch_eval(ckpt_path, task_suite, gpu_id, log_base_dir):
    p = Process(target=run_libero_eval, args=(ckpt_path, task_suite, gpu_id, log_base_dir))
    p.start()
    print(f"Starting evaluation for {ckpt_path} on GPU {gpu_id}")
    return p

def smart_schedule(ckpts, devices, task_suite, log_base_dir):
    pending = list(enumerate(ckpts))
    running = {}
    completed = []

    for gpu_id in devices:
        if pending:
            i, ckpt = pending.pop(0)
            running[gpu_id] = (launch_eval(ckpt, task_suite, gpu_id, log_base_dir), i, ckpt)

    while running or pending:
        done = []
        for gpu_id, (p, i, ckpt) in running.items():
            if not p.is_alive():
                p.join()
                done.append(gpu_id)
                completed.append((i, ckpt))
                print(f"✅ GPU {gpu_id} completed evaluation for {ckpt}")

        for gpu_id in done:
            del running[gpu_id]
            if pending:
                i, ckpt = pending.pop(0)
                running[gpu_id] = (launch_eval(ckpt, task_suite, gpu_id, log_base_dir), i, ckpt)

        if running:
            time.sleep(2)

    return completed

def main():
    parser = argparse.ArgumentParser(description="LIBERO Evaluation with GPU Scheduling")
    parser.add_argument("--dir", type=str, 
                       default="juyil/llama3.2-1B-spatial",
                       help="Parent directory containing checkpoint subdirectories")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                       help="Task suite name for evaluation")
    parser.add_argument("--devices", type=int, nargs="+", default=[0,1,2,3,4,5,6,7],
                       help="GPU device IDs to use")
    parser.add_argument("--hf_ckpts", action="store_true",
                       help="List all available checkpoints and exit")
    parser.add_argument("--log_dir", type=str, default="eval_logs",
                       help="Directory to save evaluation logs")
    
    args = parser.parse_args()
    
    if args.hf_ckpts:
        ckpts = predefined_ckpts
    else:
        ckpts = get_checkpoint_dirs(args.dir)
        if not ckpts:
            print("No valid checkpoints found.")
            return

    
    print(f"Task Suite: {args.task_suite}")
    print(f"Devices: {args.devices}")
    print(f"Checkpoints: {len(ckpts)}")
    print(f"Parent Directory: {args.dir}")
    print(f"Log Directory: {args.log_dir}")
    print("-" * 50) 
    
    t0 = datetime.now()
    
    completed = smart_schedule(ckpts, args.devices, args.task_suite, args.log_dir)
    
    t1 = datetime.now()
    
    print("=" * 50)
    print(f"✅ All evaluations finished.")
    print(f"Completed: {len(completed)} checkpoints")
    print(f"Time: {t0.strftime('%F %T')} → {t1.strftime('%F %T')} ({t1 - t0})")

if __name__ == "__main__":
    main() 
