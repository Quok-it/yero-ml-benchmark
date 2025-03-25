import torch
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import inspect
from pathlib import Path
import time
import subprocess
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_base import cuda_profiler_base
from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_mid import cuda_profiler_mid
#from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_peak import cuda_profiler_peak
from cuda_benchmarks.cuda_profiler_scripts.cuda_profiler_beyond import cuda_profiler_beyond

torch.hub.set_dir('./model_weights')

# bad_models is used for diagnosing models that require specific requirements.
bad_models = []

def get_nvidia_driver():
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

def get_nvidia_device_id(device):
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

driver_info = get_nvidia_driver()
cudnn_version = torch.backends.cudnn.version()

def run(queue, stop_event, benchmark_name, device, gpu_name, start_time):
    device_id = get_nvidia_device_id(device)
    benchmark_count = 0

    batch_sizes = [4, 16, 64, 256]
    image_sizes = [128]

    '''
    These are models that require specific image sizes to run.
    More information can be found here: https://pytorch.org/vision/stable/models.html
    '''
    if benchmark_name in ['maxvit_t', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        image_sizes = [224]
    elif benchmark_name in ['inception_v3']:
        image_sizes = [342]
    elif benchmark_name in ['vit_h_14']:
        image_sizes = [518]

    '''
    Training and Inference are considered together into a benchmark.
    There are currently three different scripts that act as a benchmark.
    In the future, there will be a fourth called _peak 
    and it uses the transformer-engine to enable FP8. 
    '''

    benchmark_total = len(batch_sizes) * len(image_sizes) * 3

    torch.manual_seed(42)
    gpu_name = gpu_name.split('\n')[0]
    for batch_size in batch_sizes:
        for image_size in image_sizes:
            transform = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,)),
            ])

            inputs = torch.randn(batch_size, 3, image_size, image_size)
            targets = torch.randint(0, 10, (batch_size,))

            inputs = torch.stack([transform(inputs[i]) for i in range(batch_size)])

            edge_model_case = False

            batch_size_str = str(batch_size)
            image_size_str = str(image_size)
            criterion = torch.nn.CrossEntropyLoss()

            for model_name in dir(models):
                if model_name == benchmark_name:
                    model_fn = getattr(models, model_name)
                    if callable(model_fn) and not isinstance(model_fn, type):
                        if "weights" in inspect.signature(model_fn).parameters:
                            try:
                                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                                queue.put('\n')
                                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loading.')

                                model = model_fn(weights='DEFAULT')
                                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                                queue.put(f'Model {model_name}, batch_size:{batch_size}, image_size:{image_size} is loaded.')
                                queue.put('\n')
                                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                                queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

                                benchmark_count += 1
                                profile_output = f"benchmarked/{gpu_name}/base/{model_name}/{batch_size_str}_{image_size_str}/"
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


                                # benchmark_count += 1
                                # profile_output = f"benchmarked/{gpu_name}/peak/{model_name}/{batch_size_str}_{image_size_str}/"
                                # path = Path(profile_output)
                                # path.mkdir(parents=True, exist_ok=True)

                                # queue.put(f'Running peak script.')
                                # edge_model_case = cuda_profiler_peak(queue, stop_event, model, device, inputs, targets,
                                #                                     criterion, optimizer, profile_output, benchmark_total,
                                #                                     benchmark_count, driver_info, batch_size, image_size,
                                #                                     device_id, start_time, cudnn_version, model_name, gpu_name)
                                # queue.put(f'Finished running peak script.')



                                benchmark_count += 1
                                profile_output = f"benchmarked/{gpu_name}/beyond/{model_name}/{batch_size_str}_{image_size_str}/"
                                path = Path(profile_output)
                                path.mkdir(parents=True, exist_ok=True)

                                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                                queue.put(f'\n')
                                queue.put(f'Running beyond script.')
                                queue.put(f'\n')
                                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                                edge_model_case = cuda_profiler_beyond(queue, stop_event, model, device, inputs, targets,
                                                                       criterion, optimizer, profile_output, benchmark_total,
                                                                       benchmark_count, driver_info, batch_size, image_size,
                                                                       device_id, start_time, cudnn_version, model_name, gpu_name)
                                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
                                queue.put(f'\n')
                                queue.put(f'Finished running beyond script.')
                                queue.put(f'\n')
                                queue.put(f'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')

                            except Exception as e:
                                print(f'Exception occur when getting {model_name} and its weight. {model_fn}\n'
                                      f'Error: {e}')
                                queue.put(f'Exception occur when getting {model_name} and its weight. {model_fn}\n'
                                      f'Error: {e}')
                                bad_models.append(model_name)
                                continue

                    if stop_event.is_set():
                        break

                    if edge_model_case == True:
                        bad_models.append(model_name)
                        edge_model_case = False

            #print(f'Found bad_models: {bad_models}')
            if stop_event.is_set():
                break

        if stop_event.is_set():
            break

    queue.put('The benchmark is finished.')
    time.sleep(5)

    if not stop_event.is_set():
        stop_event.set()
