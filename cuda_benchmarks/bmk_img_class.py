import torch
from torchvision.transforms import transforms
from torchvision.models._api import get_model
import torch.optim as optim
from pathlib import Path
import time
import subprocess
import os
from typing import Union
from torch import nn, optim as torch_optim
from multiprocessing import Queue
from multiprocessing.synchronize import Event

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_base import cuda_profiler_base
from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_mid import cuda_profiler_mid
# from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_peak import cuda_profiler_peak
from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_beyond import cuda_profiler_beyond

torch.hub.set_dir('./model_weights') # type: ignore

# bad_models is used for diagnosing models that require specific requirements.
bad_models: list[str] = []

def get_nvidia_driver() -> str:
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        output = result.stdout
        for line in output.split('\n'):
            if 'Driver Version' in line:
                driver_info = line.split('|')[1].strip()
                return driver_info
        return "NVIDIA driver version not found."

    except FileNotFoundError:
        return "nvidia-smi command not found. Make sure the NVIDIA driver is installed."

def get_nvidia_device_id(device: torch.device) -> str:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        device_ids = result.stdout.strip().split('\n')

        if not device_ids:
            return "No NVIDIA GPUs found."

        device_id = device_ids[device.index]

        return device_id

    except FileNotFoundError:
        return "nvidia-smi command not found. Make sure the NVIDIA driver is installed."

driver_info: str = get_nvidia_driver()
cudnn_version: Union[int, None] = torch.backends.cudnn.version()

def run(
    queue: Queue,
    stop_event: Event,
    model_name: str,
    device: torch.device,
    gpu_name: str,
    start_time: float
) -> None:
    device_id: str = get_nvidia_device_id(device)
    benchmark_count: int = 0

    batch_sizes: list[int] = [4, 16, 64, 256]
    image_sizes: list[int] = [128]

    if model_name in ['maxvit_t', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        image_sizes = [224]
    elif model_name in ['inception_v3']:
        image_sizes = [342]
    elif model_name in ['vit_h_14']:
        image_sizes = [518]

    benchmark_total: int = len(batch_sizes) * len(image_sizes) * 3

    torch.manual_seed(42)
    gpu_name = gpu_name.split('\n')[0]

    for batch_size in batch_sizes:
        for image_size in image_sizes:
            transform = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,)),
            ])

            inputs: torch.Tensor = torch.randn(batch_size, 3, image_size, image_size)
            targets: torch.Tensor = torch.randint(0, 10, (batch_size,))

            inputs = torch.stack([transform(inputs[i]) for i in range(batch_size)])

            edge_model_case: bool = False

            batch_size_str: str = str(batch_size)
            image_size_str: str = str(image_size)
            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

            try:
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('\n')
                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loading.')

                model: nn.Module = get_model(model_name, weights='DEFAULT')
                optimizer: torch_optim.Optimizer = optim.Adam(model.parameters(), lr=1e-3)

                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loaded.')
                queue.put('\n')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

                benchmark_count += 1

                profile_output: str = f"benchmarked/{gpu_name}/base/{model_name}/{batch_size_str}_{image_size_str}/"
                path = Path(profile_output)
                path.mkdir(parents=True, exist_ok=True)

                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                queue.put(f'\n')
                queue.put(f'Running base script.')
                queue.put(f'\n')
                edge_model_case = cuda_profiler_base(queue, stop_event, model, device, inputs, targets,
                                                     criterion, optimizer, profile_output, benchmark_total,
                                                     benchmark_count, driver_info, batch_size, image_size,
                                                     device_id, start_time, cudnn_version, model_name, gpu_name)
                queue.put(f'\n')
                queue.put(f'Finished running base script.')
                queue.put(f'\n')
                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                if edge_model_case:
                    bad_models.append(model_name)

                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('\n')
                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loading.')

                model = get_model(model_name, weights='DEFAULT')
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loaded.')
                queue.put('\n')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

                benchmark_count += 1

                profile_output = f"benchmarked/{gpu_name}/mid/{model_name}/{batch_size_str}_{image_size_str}/"
                path = Path(profile_output)
                path.mkdir(parents=True, exist_ok=True)

                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                queue.put(f'\n')
                queue.put(f'Running mid script.')
                queue.put(f'\n')
                edge_model_case = cuda_profiler_mid(queue, stop_event, model, device, inputs, targets,
                                                    criterion, optimizer, profile_output, benchmark_total,
                                                    benchmark_count, driver_info, batch_size, image_size,
                                                    device_id, start_time, cudnn_version, model_name, gpu_name)
                queue.put(f'\n')
                queue.put(f'Finished running mid script.')
                queue.put(f'\n')
                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                if edge_model_case:
                    bad_models.append(model_name)

                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('\n')
                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loading.')

                model = get_model(model_name, weights='DEFAULT')
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loaded.')
                queue.put('\n')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

                benchmark_count += 1

                profile_output = f"benchmarked/{gpu_name}/beyond/{model_name}/{batch_size_str}_{image_size_str}/"
                path = Path(profile_output)
                path.mkdir(parents=True, exist_ok=True)

                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                queue.put(f'\n')
                queue.put(f'Running beyond script.')
                queue.put(f'\n')
                edge_model_case = cuda_profiler_beyond(queue, stop_event, model, device, inputs, targets,
                                                       criterion, optimizer, profile_output, benchmark_total,
                                                       benchmark_count, driver_info, batch_size, image_size,
                                                       device_id, start_time, cudnn_version, model_name, gpu_name)
                queue.put(f'\n')
                queue.put(f'Finished running beyond script.')
                queue.put(f'\n')
                if edge_model_case:
                    bad_models.append(model_name)

            except Exception as e:
                print(f'Exception occur when getting {model_name} and its weight.\n'
                      f'Error: {e}')
                queue.put(f'Exception occur when getting {model_name} and its weight.\n'
                          f'Error: {e}')
                bad_models.append(model_name)
                continue

            if stop_event.is_set():
                break

        if stop_event.is_set():
            break

    queue.put('The benchmark is finished.')
    time.sleep(5)

    if not stop_event.is_set():
        stop_event.set()
