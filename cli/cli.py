import torch
import argparse
import multiprocessing
import time
import importlib
import subprocess
from torchvision.models._api import list_models
import torchvision.models
from tqdm import tqdm
from typing import List, Any
from torch.multiprocessing import Queue
from multiprocessing.synchronize import Event

from cuda_benchmarks import bmk_img_class

class CLI:
    def get_nvidia_device_id(self, device: torch.device) -> str:
        """
        Retrieves the UUID of a given NVIDIA GPU device.
        """
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}"

            device_ids: List[str] = result.stdout.strip().split('\n')

            if not device_ids or device_ids == ['']:
                return "No NVIDIA GPUs found."

            device_id: str = device_ids[device.index]
            return device_id

        except FileNotFoundError:
            return "nvidia-smi command not found. Make sure the NVIDIA driver is installed."
        except Exception as e:
            return f"An error occurred: {e}"


    def start_benchmark(self, queue: Queue, stop_event: Event, benchmark_file: str, model_name: str, device: torch.device, gpu_name: str) -> None: # type: ignore
        """
        Starts the benchmarking process for a given model.
        """
        start_time: float = time.time()
        benchmarking_module: Any = importlib.import_module(benchmark_file)
        benchmarking_module.run(queue, stop_event, model_name, device, gpu_name, start_time)
        stop_event.wait()


    def main(self) -> None:
        """
        Main function to run the CLI tool. 🚀
        """
        parser = argparse.ArgumentParser(description="CLI for the Yero ML Benchmark tool")
        parser.add_argument('--list-gpus', action='store_true', help='List all available GPUs')
        parser.add_argument('--list-models', action='store_true', help='List all available models')
        parser.add_argument('--model', type=str, help='The model to benchmark')
        parser.add_argument('--gpu', type=int, default=0, help='The GPU to use for the benchmark (default: 0)')
        parser.add_argument('--all', action='store_true', help='Run benchmarks on all supported models (NOT RECOMMENDED THIS COULD TAKE HOURS)')
        args: argparse.Namespace = parser.parse_args()

        if args.list_gpus:
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return

        if args.list_models:
            classification_models: List[str] = list_models(module=torchvision.models)
            for model in classification_models:
                print(model)
            return

        if not args.model and not args.all:
            print("Please specify a model to benchmark with --model <model_name> or use --all.")
            return

        if args.all:
            classification_models: List[str] = list_models(module=torchvision.models)
            for model_name in tqdm(classification_models, desc="All Models"):
                device: torch.device = torch.device(f'cuda:{args.gpu}')
                device_id: str = self.get_nvidia_device_id(device)
                # the return type for this is ignored torch, not good
                gpu_name: str = f"{torch.cuda.get_device_properties(args.gpu).name}\n{device_id}" # type: ignore
                queue: "Queue[Any]" = multiprocessing.Queue()
                stop_event: Event = multiprocessing.Event()
                bmk_img_class.run(queue, stop_event, model_name, device, gpu_name, time.time())
            return

        model_name: str = args.model
        device: torch.device = torch.device(f'cuda:{args.gpu}')
        device_id: str = self.get_nvidia_device_id(device)
        gpu_name: str = f"{torch.cuda.get_device_properties(args.gpu).name}\n{device_id}" # type: ignore
        queue: "Queue[Any]" = multiprocessing.Queue()
        stop_event: Event = multiprocessing.Event()

        bmk_img_class.run(queue, stop_event, model_name, device, gpu_name, time.time())