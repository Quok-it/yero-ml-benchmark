import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from datetime import datetime
import gc
import subprocess
import csv
import orjson
import warnings
import os

'''
Captures the current statuses of the NVIDIA GPU being used to be recorded.
'''
def device_info(queue, device, benchmark_total, benchmark_count, start_time):
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory = gpu_properties.total_memory  # Total VRAM in bytes

    free_memory = torch.cuda.memory_reserved(device)
    try:
        power_draw = torch.cuda.power_draw(device) / 1000
    except Exception as e:
        power_draw = 0
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

    power_draw =  f'{power_draw:.2f}'     # watts
    device_temp = f'{device_temp:.2f}'  # °C

    return total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str

'''
This function excludes CPU operations made from the profiler json trace file as to only contain GPU operations.
It also adds any additional information not given by the original profiler json trace file such as power draw 
and information about the benchmark.
In addition, this reduces the profiler json trace file size considerably.
'''
def convert_to_gpu_only_trace(json_path, benchmark_info, power_draw_list, free_memory_list, device_temp_list, skip_count):
    with open(json_path, "rb") as json_file:
        json_dict = orjson.loads(json_file.read())

    if 'traceEvents' in json_dict:
        json_dict['traceEvents'] = [
            entry for entry in json_dict['traceEvents'] if
            not ('cat' in entry and (entry['cat'] == 'cpu_op' or entry['cat'] == 'user_annotation'))
        ]

        found_training_opt = [True for d in json_dict['traceEvents'] if d.get('name') == 'Training Optimizer Step']
        if True in found_training_opt:
            json_dict['traceEvents'] = [
                entry for entry in json_dict['traceEvents'] if
                not ('cat' in entry and entry['cat'] == 'gpu_user_annotation' and 'Optimizer.step#' in entry['name'])
            ]
        else:
            json_dict['traceEvents'] = [dict(entry, name='Training Optimizer Step')
                                        if ('cat' in entry and entry['cat'] == 'gpu_user_annotation' and 'Optimizer.step#' in entry['name'])
                                        else entry
                                        for entry in json_dict['traceEvents']]

    json_dict['traceEvents'].extend(free_memory_list[skip_count:])
    json_dict['traceEvents'].extend(power_draw_list[skip_count:])
    json_dict['traceEvents'].extend(device_temp_list[skip_count:])

    json_dict = {'benchmark_info': benchmark_info, **json_dict}

    with open(json_path, "wb") as f:
        f.write(orjson.dumps(json_dict))

    return json_path


'''
This is the main function that runs both training and inferencing.
Both components run 105 epochs with 5 epochs being the warmup and 100 being recorded.
'''
def cuda_profiler_beyond(queue, stop_event, model, device, inputs, targets, criterion, optimizer,
                         profile_output, benchmark_total, benchmark_count, driver_info,
                         batch_size, image_size, device_id, start_time, cudnn_version, model_name, gpu_name):
    os.environ.clear()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    os.environ['TORCH_LOGS'] = '+dynamo'

    '''
    Below sets internal cuda properties to ensure maximum deterministic currently possible. 
    In addition, limitations are less restricted for NVIDIA GPUs that may have advanced hardware functionality such as tf32/bf16.
    This specific python scrypt is named cuda_profiler_beyond for the above reason, as a beyond level for a NVIDIA GPU to be at.
    Autocast and dtype=torch.float16 are used in addition to tf32
    '''
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    torch.use_deterministic_algorithms(False, warn_only=True)  # this is repeated two more times in the warning/profiler blocks

    torch.set_float32_matmul_precision('medium')

    '''
    torch._dynamo.config.suppress_errors = True  resolves issues with torch.compile() failing
    Many warnings will be shown but torch.compile will run properly
    '''
    torch._dynamo.config.suppress_errors = True

    '''
    The mode in torch.compile() can fail if the GPU fails to meet requirements set in torch.compile()
    An example is max-autotune requires at least 80 SM to run.
    Here is a link of a short discussion about it and inside is another link to the code showing the requirement.
    https://discuss.pytorch.org/t/torch-compile-warning-not-enough-sms-to-use-max-autotune-gemm-mode/184405
    '''
    compiled_mode = ''
    try:
        compiled_model = torch.compile(model, mode='max-autotune')
        compiled_mode = 'max-autotune'
    except Exception as e:
        try:
            compiled_model = torch.compile(model, mode='reduce-overhead')
            compiled_mode = 'reduce-overhead'
        except Exception as e:
            try:
                compiled_model = torch.compile(model, mode='default')
                compiled_mode = 'default'
            except Exception as e:
                compiled_model = model
                compiled_mode = 'no_compile'


    compiled_model.to(device)
    inputs, targets = inputs.to(device), targets.to(device)

    vram_limit_hit = False

    edge_model_case = False

    '''
    Below are variables to manage the profiler and how many recordings of vram/power draw/temperature needs to be skipped.
    '''
    wait_epochs = 2  # initializes profiler
    warmup_epochs = 3  # warmups profiler recording
    active_epochs = 100  # active profiler recording
    total_epochs = wait_epochs + warmup_epochs + active_epochs
    skip_count = wait_epochs + warmup_epochs

    training_rec = 3
    infer_rec = 2

    compiled_model.train()

    profile_out = profile_output

    '''
    Below grabs additional information of the NVIDIA GPU not available through torch. 
    '''
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

    nvidia_smi_version = driver_info[0].split()[0].lower()
    nvidia_smi_version_num = str(driver_info[0].split()[1]).strip().lower()

    driver_version = driver_info[1].split(':')[0].lower()
    driver_version_num = str(driver_info[1].split(':')[1]).strip().lower()

    cuda_version = driver_info[2].split(':')[0].lower()
    cuda_version_num = str(driver_info[2].split(':')[1]).strip().lower()

    benchmark_info = {
        "benchmark_type": 'training',
        "model_name": model_name,
        "batch_size": str(batch_size),
        "image_size": str(image_size),
        "cudnn_version": str(cudnn_version),
        "device_id": device_id,
        "gpu_name": gpu_name,
        nvidia_smi_version: nvidia_smi_version_num,
        driver_version: driver_version_num,
        cuda_version: cuda_version_num,
        "pcie_info": pcie_info,
        'trace_dur_unit': 'microseconds',
        'compile_mode': compiled_mode
    }

    free_memory_list = []
    power_draw_list = []
    device_temp_list = []
    total_memory_str = ''
    warning_messages = ""

    '''
    Warnings are caught and recorded in each profiler component to show any non-deterministic kernel calls made.
    '''
    queue.put("Starting Training")

    '''
    Cold start for the compiled model to compile.
    Compiling tries a few runs to make optimized graphs in place of the model.
    For more information of torch.compile(), here is a link
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    '''
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = compiled_model(inputs)


    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     with_flops=True,
                     schedule=torch.profiler.schedule(
                         wait=wait_epochs,
                         warmup=warmup_epochs,
                         active=active_epochs,
        )) as prof:
            try:
                scaler = torch.amp.GradScaler()

                for i in range(total_epochs):
                    if i >= total_epochs:
                        break
                    queue.put(f'[EPOCH]{i + 1}/{total_epochs} epochs')

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    with record_function(f"Training Forward Pass"):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = compiled_model(inputs)

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                    free_memory_list.append({"cat": "gpu_user_annotation_vram", 'name': 'Training Forward Pass VRAM Use', "dur": free_memory_str})
                    power_draw_list.append({"cat": "gpu_user_annotation_power", 'name': 'Training Forward Pass Power Use', "dur": power_draw})
                    device_temp_list.append({"cat": "gpu_user_annotation_temp", 'name': 'Training Forward Pass Current Temperature °C', "dur": device_temp})

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    with record_function(f"Training Loss Calculation"):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss = criterion(outputs, targets)

                    if stop_event.is_set():
                        break


                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                    free_memory_list.append({"cat": "gpu_user_annotation_vram", 'name': 'Training Forward Pass VRAM Use', "dur": free_memory_str})
                    power_draw_list.append({"cat": "gpu_user_annotation_power", 'name': 'Training Forward Pass Power Use', "dur": power_draw})
                    device_temp_list.append({"cat": "gpu_user_annotation_temp", 'name': 'Training Forward Pass Current Temperature °C', "dur": device_temp})

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    with record_function(f"Training Backward Pass"):
                        scaler.scale(loss).backward()

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                    free_memory_list.append({"cat": "gpu_user_annotation_vram", 'name': 'Training Loss & Backward VRAM Use', "dur": free_memory_str})
                    power_draw_list.append({"cat": "gpu_user_annotation_power", 'name': 'Training Loss & Backward Power Use', "dur": power_draw})
                    device_temp_list.append({"cat": "gpu_user_annotation_temp", 'name': 'Training Loss & Backward Current Temperature °C', "dur": device_temp})

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    with record_function(f"Training Optimizer Step"):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    if stop_event.is_set():
                        break

                    total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)

                    free_memory_list.append({"cat": "gpu_user_annotation_vram", 'name': 'Training Optimizer Step VRAM Use', "dur": free_memory_str})
                    power_draw_list.append({"cat": "gpu_user_annotation_power", 'name': 'Training Optimizer Step Power Use', "dur": power_draw})
                    device_temp_list.append({"cat": "gpu_user_annotation_temp", 'name': 'Training Optimizer Step Current Temperature °C', "dur": device_temp})

                    if free_memory > total_memory:
                        vram_limit_hit = True
                        break

                    prof.step()

            except torch.cuda.OutOfMemoryError as e:
                vram_limit_hit = True
                queue.put("NO DATA. Model exceeded GPU's VRAM.\n"
                          f"Error: {e}")

            except Exception as e:
                queue.put("NO DATA. Model has an issue with the input size or another cause. "
                          "Check model documentation for input size.\n"
                          f"Error: {e}")
                edge_model_case = True

            captured_warnings = set([str(warn_m.message) for warn_m in caught_warnings])
            warning_messages = {f'Warning {index + 1}': value for index, value in enumerate(captured_warnings)}

    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime('%Y_%m_%d-%H_%M_%S')
    profile_output = profile_out + f'{formatted_timestamp}_train.json'

    if vram_limit_hit != True:
        if edge_model_case != True:
            queue.put(f'Exporting JSON File')
            prof.export_chrome_trace(profile_output)

            benchmark_info["warnings"] = warning_messages

            profile_output = convert_to_gpu_only_trace(profile_output, benchmark_info, power_draw_list, free_memory_list, device_temp_list, skip_count * training_rec)

            queue.put(f'Finished exporting JSON File')
        else:
            with open(profile_output, "w") as f:
                f.write(f"{device_id}\nNO DATA. Model has an issue with the input size or another cause. "
                        f"Check model documentation for input size.")

    else:
        with open(profile_output, "w") as f:
            f.write(f"{device_id}\nNO DATA. Model exceeded GPU's VRAM.")

    if stop_event.is_set():
        del compiled_model
        gc.collect()
        torch.cuda.empty_cache()
        device_info(queue, device, benchmark_total, benchmark_count, start_time)

        return edge_model_case

    vram_limit_hit = False

    edge_model_case = False

    compiled_model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    compiled_model.eval()

    free_memory_list = []
    power_draw_list = []
    device_temp_list = []
    total_memory_str = ''
    warning_messages = ""

    queue.put("Starting Inference")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")


        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     with_flops=True,
                     schedule=torch.profiler.schedule(
                         wait=wait_epochs,
                         warmup=warmup_epochs,
                         active=active_epochs,
        )) as prof:
            try:
                with torch.inference_mode():
                    for i in range(total_epochs):
                        if i >= total_epochs:
                            break
                        queue.put(f'[EPOCH]{i + 1}/{total_epochs} epochs')

                        if stop_event.is_set():
                            break

                        total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)

                        if free_memory > total_memory:
                            vram_limit_hit = True
                            break

                        if stop_event.is_set():
                            break

                        with record_function(f"Inference Forward Pass"):
                            with torch.no_grad():
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    outputs = compiled_model(inputs)

                        if stop_event.is_set():
                            break

                        total_memory, free_memory, power_draw, device_temp, free_memory_str, total_memory_str = device_info(queue, device, benchmark_total, benchmark_count, start_time)
                        free_memory_list.append({"cat": "gpu_user_annotation_vram", 'name': 'Inference Forward Pass VRAM Use',"dur": free_memory_str})
                        power_draw_list.append({"cat": "gpu_user_annotation_power", 'name': 'Inference Forward Pass Power Use', "dur": power_draw})
                        device_temp_list.append({"cat": "gpu_user_annotation_temp", 'name': 'Inference Forward Pass Current Temperature °C', "dur": device_temp})

                        if free_memory > total_memory:
                            vram_limit_hit = True
                            break

                        prof.step()

            except torch.cuda.OutOfMemoryError as e:
                vram_limit_hit = True
                queue.put("NO DATA. Model exceeded GPU's VRAM.\n"
                          f"Error: {e}")

            except Exception as e:
                queue.put("NO DATA. Model has an issue with the input size or another cause. "
                          "Check model documentation for input size.\n"
                          f"Error: {e}")
                edge_model_case = True

            captured_warnings = set([str(warn_m.message) for warn_m in caught_warnings])
            warning_messages = {f'Warning {index + 1}': value for index, value in enumerate(captured_warnings)}

    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime('%Y_%m_%d-%H_%M_%S')
    profile_output = profile_out + f'{formatted_timestamp}_infer.json'

    if vram_limit_hit != True:
        if edge_model_case != True:
            queue.put(f'Exporting JSON File')
            prof.export_chrome_trace(profile_output)

            benchmark_info["warnings"] = warning_messages

            profile_output = convert_to_gpu_only_trace(profile_output, benchmark_info, power_draw_list, free_memory_list, device_temp_list, skip_count * infer_rec)

            queue.put(f'Finished exporting JSON File')

        else:
            with open(profile_output, "w") as f:
                f.write(
                    f"{device_id}\nNO DATA. Model has an issue with the input size or another cause. "
                    f"Check model documentation for input size.")
    else:
        with open(profile_output, "w") as f:
            f.write(f"{device_id}\nNO DATA. Model exceeded GPU's VRAM.")


    del compiled_model
    gc.collect()
    torch.cuda.empty_cache()
    device_info(queue, device, benchmark_total, benchmark_count, start_time)

    return edge_model_case