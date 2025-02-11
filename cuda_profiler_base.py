import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import gc

import subprocess
import csv


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# torch.set_float32_matmul_precision(“medium”)
# torch.backends.cudnn.allow_tf32 = True


def device_info(queue, device, benchmark_total, benchmark_count, start_time):
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory = gpu_properties.total_memory  # Total VRAM in bytes

    free_memory = torch.cuda.memory_reserved(device)
    power_draw = torch.cuda.power_draw(device) / 1000
    device_temp = torch.cuda.temperature(device)

    since_time = time.time() - start_time

    hours = int(since_time // 3600)  # 3600 seconds in an hour
    minutes = int((since_time % 3600) // 60)  # Remaining minutes
    seconds = int(since_time % 60)  # Remaining seconds

    queue.put(f'[INFO]{device_temp:.2f} °C GPU Temperature\n\n'
              f'{power_draw:.2f} Watts in Use\n\n'
              f'{free_memory / (1024 ** 3):.2f} GB / {total_memory / (1024 ** 3):.2f} GB VRAM in Use\n\n'
              f'{benchmark_count}/{benchmark_total} Benchmarks\n\n'
              f'Time Since Starting: {hours:02}:{minutes:02}:{seconds:02}')

    free_memory_str = f'{free_memory / (1024 ** 3):.2f}'
    total_memory_str = f'{total_memory / (1024 ** 3):.2f}'

    power_draw =  f'{power_draw:.2f}'
    device_temp = f'{device_temp:.2f}°C'

    return total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str



def cuda_profiler_base(queue, stop_event, model, device, train_loader, test_loader, criterion, optimizer,
                         profile_output, benchmark_total, benchmark_count, driver_info,
                         batch_size, image_size, device_id, start_time, cudnn_version, model_name, gpu_name):

    model.to(device)

    vram_limit_hit = False

    edge_model_case = False

    model.train()

    profile_out = profile_output + '_' + str(batch_size) + '_' + str(image_size)

    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current',
         '--format=csv'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    pcie_info = ''
    if result.returncode == 0:
        reader = csv.DictReader(result.stdout.splitlines(), delimiter=',')
        for idx, row in enumerate(reader):
            if str(device.index) == row['index']:
                pcie_info = row[" pcie.link.gen.current"].strip() + 'x' + row[" pcie.link.width.current"].strip()
    else:
        print("Error running nvidia-smi:", result.stderr)
        pcie_info = result.stderr

    driver_info = [item.strip() for item in driver_info.split('  ') if item.strip()]

    nvidia_smi_version = driver_info[0].split()[0]
    nvidia_smi_version_num = str(driver_info[0].split()[1]).strip()

    driver_version = driver_info[1].split(':')[0]
    driver_version_num = str(driver_info[1].split(':')[1]).strip()

    cuda_version = driver_info[2].split(':')[0]
    cuda_version_num = str(driver_info[2].split(':')[1]).strip()


    free_memory_list = []
    power_draw_list = []
    device_temp_list = []
    total_memory_str = ''

    queue.put("Starting Training")
    with profile(activities=[torch.profiler.ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_flops=True, schedule=torch.profiler.schedule(
        wait=2,
        warmup=3,  # Skip the first 3 iterations of profiling data collection
        active=10,  # Start profiling from iteration 4 to iteration 8
    )) as prof:
        try:
            for i, (inputs, targets) in enumerate(train_loader):
                if i >= 15:  # Limit profiling to 30 batches
                    break
                queue.put(f'[EPOCH]{i + 1}/15 epochs')

                if stop_event.is_set():
                    break

                inputs, targets = inputs.to(device), targets.to(device)

                if stop_event.is_set():
                    break

                total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                free_memory_list.append(free_memory_str)
                power_draw_list.append(power_draw)
                device_temp_list.append(device_temp)

                if free_memory > total_memory:
                    vram_limit_hit = True
                    break

                # print(f'Running Forward Pass')
                #queue.put(f'Running Training Forward Pass')
                with record_function("Training Forward Pass"):
                    outputs = model(inputs)

                if stop_event.is_set():
                    break


                total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                free_memory_list.append(free_memory_str)
                power_draw_list.append(power_draw)
                device_temp_list.append(device_temp)

                if free_memory > total_memory:
                    vram_limit_hit = True
                    break

                # print(f'Running Loss & Backward')
                #queue.put(f'Running Training Loss & Backward')
                with record_function("Training Loss & Backward"):
                    loss = criterion(outputs, targets)
                    loss.backward()

                if stop_event.is_set():
                    break

                total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                free_memory_list.append(free_memory_str)
                power_draw_list.append(power_draw)
                device_temp_list.append(device_temp)

                if free_memory > total_memory:
                    vram_limit_hit = True
                    break

                # print(f'Running Optimizer Step')
                #queue.put(f'Running Training Optimizer Step')
                with record_function("Training Optimizer Step"):
                    optimizer.step()
                    optimizer.zero_grad()

                if stop_event.is_set():
                    break

                total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)

                free_memory_list.append(free_memory_str)
                power_draw_list.append(power_draw)
                device_temp_list.append(device_temp)

                if free_memory > total_memory:
                    vram_limit_hit = True
                    break

                prof.step()

        except torch.cuda.OutOfMemoryError as e:
            vram_limit_hit = True
            print("NO DATA. Model exceeded GPU's VRAM")
            print(e)
            queue.put("NO DATA. Model exceeded GPU's VRAM")

        except Exception as e:
            print("NO DATA. Model has an issue with the input size or another cause. Check model documentation for input size.")
            print(e)
            queue.put("NO DATA. Model has an issue with the input size or another cause. Check model documentation for input size.")
            edge_model_case = True

        free_memory_str = ' '.join(free_memory_list)
        power_draw_str = ' '.join(power_draw_list)
        device_temp_str = ' '.join(device_temp_list)

        prof.add_metadata("Benchmark Type", 'Training')
        prof.add_metadata("model_name", model_name)
        prof.add_metadata("batch_size", str(batch_size))
        prof.add_metadata("image_size", str(image_size))
        prof.add_metadata("image_size", str(image_size))
        prof.add_metadata("cudnn_version", str(cudnn_version))
        prof.add_metadata("device_id", device_id)
        prof.add_metadata("gpu_name", gpu_name)
        prof.add_metadata(nvidia_smi_version, nvidia_smi_version_num)
        prof.add_metadata(driver_version, driver_version_num)
        prof.add_metadata(cuda_version, cuda_version_num)
        prof.add_metadata("pcie_info", pcie_info)
        prof.add_metadata("Total VRAM", total_memory_str)
        prof.add_metadata("Free Memory Time Period", free_memory_str)
        prof.add_metadata("Power Draw Time Period", power_draw_str)
        prof.add_metadata("Device Temperature Time Period", device_temp_str)


    profile_output = profile_out + '_profiler_basic_tr.json'


    if vram_limit_hit != True:
        if edge_model_case != True:
            queue.put(f'Exporting JSON File')
            prof.export_chrome_trace(profile_output)
            # Print the profiler results
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            with open(profile_output, "w") as f:
                f.write(f"{device_id}\nNO DATA. Model has an issue with the input size or another cause. Check model documentation for input size.")

    else:
        with open(profile_output, "w") as f:
            f.write(f"{device_id}\nNO DATA. Model exceeded GPU's VRAM.")



    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    # Inference
    ############################################################################################################
    ############################################################################################################

    if stop_event.is_set():
        del model
        gc.collect()
        torch.cuda.empty_cache()
        device_info(queue, device, benchmark_total, benchmark_count, start_time)
        # time.sleep(1)

        return edge_model_case

    vram_limit_hit = False

    edge_model_case = False

    model.eval()

    free_memory_list = []
    power_draw_list = []
    device_temp_list = []
    total_memory_str = ''
    queue.put("Starting Inference")
    with profile(activities=[torch.profiler.ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_flops=True, schedule=torch.profiler.schedule(
        wait=2,
        warmup=3,  # Skip the first 3 iterations of profiling data collection
        active=10,  # Start profiling from iteration 4 to iteration 8
    )) as prof:
        try:
            for i, (inputs, targets) in enumerate(test_loader):
                with torch.inference_mode():
                    if i >= 15:  # Limit profiling to 5 batches
                        break

                    queue.put(f'[EPOCH]{i + 1}/15 epochs')

                    if stop_event.is_set():
                        break

                    inputs, targets = inputs.to(device), targets.to(device)

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                    free_memory_list.append(free_memory_str)
                    power_draw_list.append(power_draw)
                    device_temp_list.append(device_temp)

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    if stop_event.is_set():
                        break

                    # print(f'Running Forward Pass')
                    #queue.put(f'Running Inference Forward Pass')
                    with record_function("Inference Forward Pass"):
                        outputs = model(inputs)

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                    free_memory_list.append(free_memory_str)
                    power_draw_list.append(power_draw)
                    device_temp_list.append(device_temp)

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    #queue.put(f'Running Inference torch.max')
                    with record_function("Inference torch.max"):
                        _, predicted_class = torch.max(outputs, 1)

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total,
                                                                                     benchmark_count, start_time)
                    free_memory_list.append(free_memory_str)
                    power_draw_list.append(power_draw)
                    device_temp_list.append(device_temp)

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    prof.step()

        except torch.cuda.OutOfMemoryError as e:
            vram_limit_hit = True
            print("NO DATA. Model exceeded GPU's VRAM")
            print(e)
            queue.put("NO DATA. Model exceeded GPU's VRAM")

        except Exception as e:
            print(
                "NO DATA. Model has an issue with the input size or another cause. Check model documentation for input size.")
            print(e)
            queue.put(
                "NO DATA. Model has an issue with the input size or another cause. Check model documentation for input size.")
            edge_model_case = True

        free_memory_str = ' '.join(free_memory_list)
        power_draw_str = ' '.join(power_draw_list)
        device_temp_str = ' '.join(device_temp_list)

        prof.add_metadata("Benchmark Type", 'Inference')
        prof.add_metadata("model_name", model_name)
        prof.add_metadata("batch_size", str(batch_size))
        prof.add_metadata("image_size", str(image_size))
        prof.add_metadata("image_size", str(image_size))
        prof.add_metadata("cudnn_version", str(cudnn_version))
        prof.add_metadata("device_id", device_id)
        prof.add_metadata("gpu_name", gpu_name)
        prof.add_metadata(nvidia_smi_version, nvidia_smi_version_num)
        prof.add_metadata(driver_version, driver_version_num)
        prof.add_metadata(cuda_version, cuda_version_num)
        prof.add_metadata("pcie_info", pcie_info)
        prof.add_metadata("Total VRAM", total_memory_str)
        prof.add_metadata("Free Memory Time Period", free_memory_str)
        prof.add_metadata("Power Draw Time Period", power_draw_str)
        prof.add_metadata("Device Temperature Time Period", device_temp_str)

    profile_output = profile_out + '_profiler_basic_in.json'


    if vram_limit_hit != True:
        if edge_model_case != True:
            queue.put(f'Exporting JSON File')
            prof.export_chrome_trace(profile_output)
            # Print the profiler results
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        else:
            with open(profile_output, "w") as f:
                f.write(
                    f"{device_id}\nNO DATA. Model has an issue with the input size or another cause. Check model documentation for input size.")
    else:
        with open(profile_output, "w") as f:
            f.write(f"{device_id}\nNO DATA. Model exceeded GPU's VRAM.")


    del model
    gc.collect()
    torch.cuda.empty_cache()
    device_info(queue, device, benchmark_total, benchmark_count, start_time)
    #time.sleep(1)

    return edge_model_case