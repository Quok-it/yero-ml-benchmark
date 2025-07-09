import torch
import multiprocessing
import threading
import customtkinter as ctk
import time
import subprocess
import importlib
from torchvision.models import list_models
import torchvision
import os
from customtkinter.windows.widgets.ctk_frame import CTkFrame
from tkinter import Event
from typing import Optional, Union

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

lock = threading.Lock()

font_type = {'family': 'Consolas', 'size': 14}

def get_nvidia_device_id(device: torch.device) -> str:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        device_ids = result.stdout.strip().split('\n')

        if not device_ids:
            return "No NVIDIA GPUs found."

        device_id = device_ids[device.index]

        return device_id

    except Exception as e:
        return f"nvidia-smi command not found. Make sure the NVIDIA driver is installed. Error: {e}"

def start_benchmark(queue, stop_event, benchmark_file, model_name, device, gpu_name):
    start_time = time.time()

    benchmarking = importlib.import_module(benchmark_file)
    benchmarking.run(queue, stop_event, model_name, device, gpu_name, start_time)

    stop_event.wait()

class MyActionFrame(ctk.CTkFrame):
    def __init__(self, master: CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.run_button = ctk.CTkButton(self,
                                        text="Run",
                                        command=self.run_benchmark,
                                        border_color='#FFB000',
                                        border_width=3,
                                        font=(font_type['family'], font_type['size']),
                                        text_color="#FFB000",
                                        fg_color="#141414")
        self.run_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        self.end_button = ctk.CTkButton(self,
                                        text="End",
                                        command=self.end_benchmark,
                                        border_color='#FFB000',
                                        border_width=3,
                                        font=(font_type['family'], font_type['size']),
                                        text_color="#FFB000",
                                        fg_color="#141414")
        self.end_button.grid(row=1, column=2, padx=10, pady=10, sticky="w")


        self.mygpuframe = MyGPUFrame(master=master,
                                     border_color='#141414',
                                     border_width=0,
                                     fg_color='#141414')
        self.mygpuframe.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.mybenchmarkframe = MyBenchmarkFrame(master=master,
                                                 border_color="#141414",
                                                 border_width=0,
                                                 fg_color="#141414")
        self.mybenchmarkframe.grid(row=0, column=1, padx=10, columnspan=2, pady=10, sticky="nsew")

        self.mystatusframe = MyStatusFrame(master=master,
                                           border_color='#141414',
                                           border_width=0,
                                           fg_color='#141414')
        self.mystatusframe.grid(row=1, column=0, padx=10, columnspan=3, rowspan=2, pady=10, sticky="nsew")

        self.mycurrentframe = MyCurrentFrame(master=master,
                                             border_color='#141414',
                                             border_width=0,
                                             fg_color='#141414')
        self.mycurrentframe.grid(row=1, column=3, padx=10, rowspan=2, pady=10, sticky="nsew")


        self.running = False

        self.processes = []
        self.process = None
        self.listener = None

        multiprocessing.set_start_method('spawn')
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

        self.threads = []

    '''
    listen_for_messages displays text in the GUI from the queue from multiprocessing. 
    '''
    def listen_for_messages(self, queue, stop_event):
        while not stop_event.is_set():
            if not queue.empty():
                message = queue.get_nowait()
                with lock:
                    if stop_event.is_set():
                        break
                    try:
                        cut_message = message.split(']')[0]
                        match cut_message:
                            case '[INFO':
                                info_message = message.split(']')[1]
                                self.mycurrentframe.update_info(text=info_message)
                            case '[EPOCH':
                                text_update_epoch = message.split(']')[1]
                                self.mystatusframe.update_status(text=text_update_epoch)
                            case _:
                                text_update = f"{message}"
                                self.mystatusframe.update_status(text=text_update)

                    except Exception as e:
                        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                        print(f'Exception has occur: {e}')
                        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                        continue

    def run_benchmark(self):
        model_name = self.mybenchmarkframe.get().split('~')[0]
        benchmark_file = self.mybenchmarkframe.get().split('~')[1]
        gpu_num = self.mygpuframe.get()
        cuda_gpus_count = self.mygpuframe.cuda_gpus_count

        device = ''
        if gpu_num < cuda_gpus_count:
            device = torch.device(f'cuda:{str(gpu_num)}')
            device_id = get_nvidia_device_id(device)
            gpu_name = torch.cuda.get_device_properties(gpu_num).name + "\n" + device_id
        else:
            gpu_name = 'GPU DEVICE'

        if model_name != "":

            if self.running != True:

                self.running = True
                self.mystatusframe.update_status(text=f"==============================================================="
                                                      f"===============================================================\n"
                                                      f"==============================================================="
                                                      f"===============================================================")
                self.mystatusframe.update_status(text=f"\n")
                self.mystatusframe.update_status(text=f"Running {model_name} benchmark for\n{gpu_name}")
                self.mystatusframe.update_status(text=f"\n")
                self.mystatusframe.update_status(text=f"==============================================================="
                                                      f"===============================================================\n"
                                                      f"==============================================================="
                                                      f"===============================================================")

                self.listener = threading.Thread(target=self.listen_for_messages,
                                                 args=(self.queue,
                                                       self.stop_event),
                                                 daemon=True)
                self.threads.append(self.listener)
                self.listener.start()

                self.process = multiprocessing.Process(target=start_benchmark,
                                                       args=(self.queue,
                                                             self.stop_event,
                                                             benchmark_file,
                                                             model_name,
                                                             device,
                                                             gpu_name),
                                                       daemon=True)
                self.processes.append(self.process)
                self.process.start()

            else:
                self.mystatusframe.update_status(text=f"Benchmark is active for {gpu_name}. "
                                                      f"Click on the End button to run a new benchmark.")
        else:
            self.mystatusframe.update_status(text="Select a benchmark to run.")
        return

    def end_benchmark(self):

        if self.running == True:
            self.mystatusframe.update_status(text=f"WMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWM")
            self.mystatusframe.update_status(text=f"WMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWM")
            self.mystatusframe.update_status(text=f"Ending the active benchmark.")
            self.mystatusframe.update_status(text=f"Please wait as processes finish to avoid any data corruption.")
            self.running = False

            self.stop_event.set()

            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
            self.listener.join()

            self.queue = multiprocessing.Queue()
            self.stop_event = multiprocessing.Event()

            self.mystatusframe.update_status(text=f"The active benchmark has ended.")
            self.mystatusframe.update_status(text=f"WMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWM")
            self.mystatusframe.update_status(text=f"WMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWM")

        else:
            self.mystatusframe.update_status(text="No benchmark is active. Select a benchmark to run.")
        return

    def on_close(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=1)

        for p in self.processes:
            p.join(timeout=1)
            p.terminate()

class MyGPUFrame(ctk.CTkFrame):
    def __init__(self, master: CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.gpus = []
        self.gpu_names = []
        self.cuda_gpus_count = 0
        self.variable = ctk.IntVar()

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.my_gpu_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                         fg_color="#141414",
                                         border_color='#141414',
                                         border_width=0)

        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{str(i)}')
            device_id = get_nvidia_device_id(device)
            gpu_name = torch.cuda.get_device_properties(i).name + "\n" + device_id
            radio_gpu = ctk.CTkRadioButton(self.my_gpu_frame,
                                           text=gpu_name,
                                           value=i,
                                           variable=self.variable,
                                           text_color='#FFB000',
                                           font=(font_type['family'], font_type['size']))
            radio_gpu.grid(row=i, column=0, padx=10, pady=10, sticky="w")
            self.gpus.append(radio_gpu)
            self.gpu_names.append(gpu_name)
            self.cuda_gpus_count += 1

        self.variable.set(0)

        self.scrollable_canvas.add_content(self.my_gpu_frame, row=0, column=0)

    def get(self) -> int:
        return self.variable.get()

    def get_selected_gpu_name(self):
        gpu_name = self.gpu_names[self.get()]
        return gpu_name

class MyBenchmarkFrame(ctk.CTkFrame):
    def __init__(self, master: CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.benchmark = []
        self.variable = ctk.StringVar(value="")

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.my_benchmark_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                               fg_color="#141414")

        model_count = 0
        classification_models = list_models(module=torchvision.models)
        for model_name in classification_models:
            bench_model = ctk.CTkRadioButton(self.my_benchmark_frame,
                                             text=model_name,
                                             value=f"{model_name}~cuda_benchmarks.bmk_img_class",
                                             variable=self.variable,
                                             text_color='#FFB000',
                                             font=(font_type['family'], font_type['size']))
            bench_model.grid(row=model_count, column=0, padx=10, pady=10, sticky="nsew")
            model_count += 1
            self.benchmark.append(bench_model)

        self.scrollable_canvas.add_content(self.my_benchmark_frame, row=0, column=0)

    def get(self) -> str:
        return self.variable.get()

class MyStatusFrame(ctk.CTkFrame):
    def __init__(self, master: CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.status = ctk.CTkTextbox(self,
                                     fg_color="#141414",
                                     text_color="#FFB000",
                                     border_color='#FFB000',
                                     border_width=3)
        self.status.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def update_status(self, text: str):
        if text != '':
            self.status.insert(index=self.status.index("end"), text=text + "\n")
            self.status.see(index=self.status.index("end"))
        return

class MyCurrentFrame(ctk.CTkFrame):
    def __init__(self, master: CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.info = ctk.CTkTextbox(self,
                                   fg_color="#141414",
                                   text_color="#FFB000",
                                   border_color='#FFB000',
                                   border_width=3)
        self.info.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    def update_info(self, text):
        self.info.delete("0.0", "end")
        self.info.insert(index="0.0", text=text)
        return

class ScrollableCanvas(ctk.CTkFrame):
    def __init__(self, master: Optional[Union[MyBenchmarkFrame, MyGPUFrame]]=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.canvas = ctk.CTkCanvas(self,
                                    bg='#141414',
                                    highlightbackground='#141414',
                                    highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scroll = ctk.CTkScrollbar(self,
                                         orientation="vertical",
                                         command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.h_scroll = ctk.CTkScrollbar(self,
                                         orientation="horizontal",
                                         command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=self.v_scroll.set,
                              xscrollcommand=self.h_scroll.set)

        self.content_frame = ctk.CTkFrame(self.canvas,
                                          border_color='#FFB000',
                                          border_width=3,
                                          fg_color='#141414')
        self.content_frame.grid(row=0, column=0, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.create_window((0, 0),
                                  window=self.content_frame,
                                  anchor="nw")
        self.min_width = -1
        self.min_height = -1

        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event: Event):
        if self.min_width != -1:
            max_width = max(self.canvas.winfo_width(), self.min_width)
            max_height = max(self.canvas.winfo_height(), self.min_height)

            self.canvas.create_window((0, 0),
                                      window=self.content_frame,
                                      anchor="nw",
                                      height=max_height,
                                      width=max_width)

    def add_content(self, content: CTkFrame, row: int, column: int, rowspan: int=1, columnspan: int=1, padx: int=10, pady: int=10, sticky: str="nsew"):
        content.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky)

        self.content_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.update_idletasks()

        self.min_width = max(self.canvas.winfo_width(), self.content_frame.winfo_width())
        self.min_height = max(self.canvas.winfo_height(), self.content_frame.winfo_height())
