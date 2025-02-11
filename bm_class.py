import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import inspect
from pathlib import Path
import time
import subprocess

from cuda_profiler_base import cuda_profiler_base


torch.hub.set_dir('model_weights')


batch_sizes = [16, 32, 64, 128, 256]
image_sizes = [128, 256, 224, 299, 518]



bad_models = []
vram_issues = []

model_total = 0
for model_name in dir(models):
    model_fn = getattr(models, model_name)
    if callable(model_fn) and not isinstance(model_fn, type):
        if "weights" in inspect.signature(model_fn).parameters:
            model_total += 1

benchmark_total = len(batch_sizes) * len(image_sizes) * model_total
benchmark_count = 0


def get_nvidia_driver():
    try:
        # Run the `nvidia-smi` command to get driver version
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors in the command execution
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Parse the output to get the driver version
        output = result.stdout
        for line in output.split('\n'):
            if 'Driver Version' in line:
                driver_info = line.split('|')[1].strip()
                return driver_info
        return "NVIDIA driver version not found."

    except FileNotFoundError:
        return "nvidia-smi command not found. Make sure the NVIDIA driver is installed."


# Example usage
driver_info = get_nvidia_driver()
#print(f"Current NVIDIA Driver Version: {driver_info}")
cudnn_version = torch.backends.cudnn.version()
#print(f"Current cuDNN Version: {cudnn_version}")

def get_nvidia_device_id(device):
    try:
        # Run the `nvidia-smi` command
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors in the command execution
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Get the GPU indices (device IDs)
        device_ids = result.stdout.strip().split('\n')

        if not device_ids:
            return "No NVIDIA GPUs found."

        device_id = device_ids[device.index]

        return device_id  # This will return a list of device IDs

    except FileNotFoundError:
        return "nvidia-smi command not found. Make sure the NVIDIA driver is installed."


# Example usage




def run(queue, stop_event, device, gpu_name, timestamp, start_time):
    device_id = get_nvidia_device_id(device)
    print(f"Device IDs: {device_id}")
    global benchmark_count
    for batch_size in batch_sizes:
        for image_size in image_sizes:
            # Define the transformation (e.g., converting to tensor and normalizing)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Resize(image_size)
            ])

            # Download and load the MNIST dataset
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

            # Create DataLoader for batching and shuffling the data
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # # Now you can iterate over the train_loader or test_loader for training/testing
            # for images, labels in train_loader:
            #     print(images.shape, labels.shape)  # Example: Shape of the batch of images and labels
            #     break  # Just printing the first batch

            #model = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(use_device)


            # print(f"\n[Profiling ResNet-50 on {use_device} using torch.profiler...]")
            # model.to(use_device)
            edge_model_case = False
            for model_name in dir(models):
                model_fn = getattr(models, model_name)
                if callable(model_fn) and not isinstance(model_fn, type):
                    if "weights" in inspect.signature(model_fn).parameters:
                        benchmark_count += 1
                        try:
                            queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            print(f'Model: {model_name} is loading.')
                            queue.put(f'Model {model_name} is loading.')
                            model = model_fn(weights='DEFAULT')
                            print(f'Model: {model_name} is loaded.')
                            queue.put(f'Model {model_name} is loaded.')
                            queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            queue.put('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

                            # call trainer profiler script, pass in model, queue, stop_event, other things like transform, dataset, gpu etc.
                            # the model inside checks for the stop event too at key checkpoints
                            path = Path(f"Classification/{timestamp}/")
                            path.mkdir(parents=True, exist_ok=True)
                            profile_output = f"Classification/{timestamp}/{gpu_name}_{model_name}"

                            criterion = torch.nn.CrossEntropyLoss()
                            optimizer = optim.Adam(model.parameters(), lr=1e-3)
                            gpu_name = gpu_name.split('_')[0]
                            time.sleep(1)
                            print('STARTING THE TRAINER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            edge_model_case = cuda_profiler_base(queue, stop_event, model, device, train_loader, test_loader, criterion, optimizer,
                                                                   profile_output, benchmark_total, benchmark_count, driver_info,
                                                                   batch_size, image_size, device_id, start_time, cudnn_version, model_name, gpu_name)

                        except Exception as e:
                            print(f'Exception occur when getting {model_name} and its weight. {model_fn}')
                            queue.put(f'Exception occur when getting {model_name} and its weight. {model_fn}')
                            print(e)
                            bad_models.append(model_name)
                            continue
                        # else:
                        #     print(f'{model_name} requires specific tensor shape to be used. {model_fn}')
                        #     queue.put(f'{model_name} requires specific tensor shape to be used. {model_fn}')
                        #     bad_models.append(model_name)


                if stop_event.is_set():
                    break

                if edge_model_case == True:
                    bad_models.append(model_name)
                    edge_model_case = False


            print('BAD MODELS HERE')
            print(bad_models)
            if stop_event.is_set():
                break
        print('last for loop')
        if stop_event.is_set():
            break
    print('in the def region')
    if not stop_event.is_set():
        stop_event.set()
    print('last in the def region')
