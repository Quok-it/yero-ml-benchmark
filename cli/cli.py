import logging
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

from cli.export_results import Results
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

    def _run_benchmark(self, args: argparse.Namespace):
            if args.list_gpus:
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                return

            if args.list_models:
                print("\n".join(list_models(module=torchvision.models)))
                return

            if not args.model and not args.all:
                print("Please specify a model with --model <model_name> or use --all.")
                return

            models_to_benchmark = list_models(module=torchvision.models) if args.all else [args.model]
            device = torch.device(f'cuda:{args.gpu}')
            device_id = self.get_nvidia_device_id(device)
            gpu_name = f"{torch.cuda.get_device_properties(args.gpu).name}\n{device_id}"

            for model_name in tqdm(models_to_benchmark, desc="Benchmarking Models"):
                logging.info(f"Running benchmark: {model_name} on {torch.cuda.get_device_name(args.gpu)}")
                queue: Queue = multiprocessing.Queue()
                stop_event: Event = multiprocessing.Event()
                bmk_img_class.run(queue, stop_event, model_name, device, gpu_name, time.time())
    

    def main(self) -> None:
        """
        Main function to run the CLI tool.
        """
        parser = argparse.ArgumentParser(description="CLI for the Yero ML Benchmark tool")
        subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

        parser_run = subparsers.add_parser('run', help='Run a new benchmark.')
        parser_run.add_argument('--list-gpus', action='store_true', help='List all available GPUs')
        parser_run.add_argument('--list-models', action='store_true', help='List all available models')
        parser_run.add_argument('--model', type=str, help='The model to benchmark')
        parser_run.add_argument('--gpu', type=int, default=0, help='The GPU to use for the benchmark (default: 0)')
        parser_run.add_argument('--all', action='store_true', help='Run benchmarks on all supported models (NOT RECOMMENDED THIS COULD TAKE HOURS)')
        parser_run.set_defaults(func=self._run_benchmark)

        parser_results = subparsers.add_parser('results', help='Analyze existing benchmark results.')
        Results(parser_results)
        
        args: argparse.Namespace = parser.parse_args()
        args.func(args)

