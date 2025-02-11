import customtkinter as ctk
import torch
import multiprocessing
import threading

import time
from datetime import datetime
import subprocess

import importlib

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def start_benchmark(queue, stop_event, benchmark_run, device, gpu_name):
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime('%Y_%m_%d-%H_%M_%S')
    start_time = time.time()

    benchmarking = importlib.import_module(benchmark_run)
    benchmarking.run(queue, stop_event, device, gpu_name, formatted_timestamp, start_time)


class MyGPUFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title = "Select a GPU"
        self.grid_columnconfigure(0, weight=1)

        self.gpus = []
        self.variable = ctk.IntVar()

        for i in range(torch.cuda.device_count()):
            # print(torch.cuda.get_device_properties(i).name)
            gpu_name = torch.cuda.get_device_properties(i).name + "_" + str(i)
            radio_gpu = ctk.CTkRadioButton(self, text=gpu_name, value=i, variable=self.variable)
            radio_gpu.grid(row=i, column=0, padx=10, pady=10, sticky="w")
            self.gpus.append(radio_gpu)

        self.variable.set(0)

    def get(self):
        return self.variable.get()

class MyBenchmarkFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title = "Select A Benchmark To Run"
        self.grid_columnconfigure(0, weight=1)

        self.benchmark = []
        self.variable = ctk.StringVar(value="")

        bench_1 = ctk.CTkRadioButton(self, text="Classification", value="bm_class", variable=self.variable)
        bench_1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.benchmark.append(bench_1)

    def get(self):
        return self.variable.get()

class MyStatusFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.status = ctk.CTkTextbox(self)
        self.status.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.status.configure(state="disabled", wrap="word")


    def update_status(self, text):
        if text != '':
            self.status.configure(state="normal", wrap="word")
            self.status.insert(index=self.status.index("end"), text=text + "\n")
            self.status.see(index=self.status.index("end"))
            self.status.configure(state="disabled", wrap="word")
        return


class MyCurrentFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.info = ctk.CTkTextbox(self)
        self.info.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.info.configure(state="disabled")

        # self.vram = ctk.CTkTextbox(self)
        # self.vram.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        # self.vram.configure(state="disabled")
        #
        # self.temp = ctk.CTkTextbox(self)
        # self.temp.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        # self.temp.configure(state="disabled")

    def update_info(self, text):
        self.info.configure(state="normal")
        self.info.delete("0.0", "end")
        self.info.insert(index="0.0", text=text)
        self.info.configure(state="disabled")
        return

    # def update_pwr(self, text):
    #     self.watts.configure(state="normal")
    #     self.watts.delete("0.0", "end")
    #     self.watts.insert(index="0.0", text=text)
    #     self.watts.configure(state="disabled")
    #     return
    #
    # def update_vram(self, text):
    #     self.vram.configure(state="normal")
    #     self.vram.delete("0.0", "end")
    #     self.vram.insert(index="0.0", text=text)
    #     self.vram.configure(state="disabled")
    #     return
    #
    # def update_temp(self, text):
    #     self.temp.configure(state="normal")
    #     self.temp.delete("0.0", "end")
    #     self.temp.insert(index="0.0", text=text)
    #     self.temp.configure(state="disabled")
    #     return

class MyActionFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title = "Click to Run or End"
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.run_button = ctk.CTkButton(self, text="Run", command=self.run_benchmark)
        self.run_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.end_button = ctk.CTkButton(self, text="End", command=self.end_benchmark)
        self.end_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        #######################
        #######################
        self.mygpuframe = MyGPUFrame(master=master, border_color='white', border_width=5, fg_color='gray')
        self.mygpuframe.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.mybenchmarkframe = MyBenchmarkFrame(master=master, border_color="green", border_width=5,
                                                 fg_color="blue")
        self.mybenchmarkframe.grid(row=0, column=1, padx=10, columnspan=2, pady=10, sticky="nsew")

        self.mystatusframe = MyStatusFrame(master=master, border_color='white', border_width=5,
                                           fg_color='orange')
        self.mystatusframe.grid(row=1, column=0, padx=10, columnspan=3, rowspan=2, pady=10, sticky="nsew")

        self.mycurrentframe = MyCurrentFrame(master=master, border_color='white', border_width=5,
                                             fg_color='gray')
        self.mycurrentframe.grid(row=1, column=3, padx=10, rowspan=2, pady=10, sticky="nsew")


        self.running = False

        # self.checkbox_1 = ctk.CTkCheckBox(self, text="checkbox 1")
        # self.checkbox_1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        # self.checkbox_2 = ctk.CTkCheckBox(self, text="checkbox 2")
        # self.checkbox_2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.processes = []
        self.process = None
        self.listener = None
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

        self.threads = []

    def listen_for_messages(self, queue, stop_event):
        while not stop_event.is_set():
            if not queue.empty():
                message = queue.get()
                try:
                    cut_message = message.split(']')[0]
                    match cut_message:
                        case '[INFO':
                            info_message = message.split(']')[1]
                            self.mycurrentframe.update_info(text=info_message)
                        case '[EPOCH':
                            #print(f"{message}")
                            text_update_epoch = message.split(']')[1]
                            self.mystatusframe.update_status(text=text_update_epoch)
                        case _:
                            #print(f"{message}")
                            text_update = f"{message}"
                            self.mystatusframe.update_status(text=text_update)

                except Exception as e:
                    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                    print(f'Exception has occur: {e}')
                    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
                    continue

            #time.sleep(0.01)  # Prevent CPU overuse

    # def listen_for_info(self, queue, stop_event):
    #
    #     while not stop_event.is_set():
    #         if not queue.empty():
    #             message = queue.get()
    #             if message.split(']')[0] == '[VRAM':
    #                 vram_message = message.split(']')[1]
    #                 self.mycurrentframe.update_vram(text=vram_message)
    #
    #
    #         time.sleep(1)  # Prevent CPU overuse



            #self.mycurrentframe.update_watts(text=text_watts)


    def run_benchmark(self):
        benchmark_name = self.mybenchmarkframe.get()
        gpu_num = self.mygpuframe.get()
        gpu_name = torch.cuda.get_device_properties(gpu_num).name + "_" + str(gpu_num)

        if benchmark_name != "":
            if self.running != True:

                self.running = True
                self.mystatusframe.update_status(text=f"running {benchmark_name} benchmark for {gpu_name}")
                print(f"running {benchmark_name} benchmark for {gpu_name}")
                # Call function to start the benchmark with the agruments above


                # Start the worker process
                # self.process = multiprocessing.Process(target=worker.run, args=(self.queue, self.stop_event))
                # self.processes.append(self.process)
                # self.process.start()
                benchmark_run = self.mybenchmarkframe.get()
                device = torch.device(f'cuda:{str(self.mygpuframe.get())}')



                self.process = multiprocessing.Process(target=start_benchmark, args=(self.queue, self.stop_event, benchmark_run, device, gpu_name), daemon=True)
                self.processes.append(self.process)
                self.process.start()


                #self.listener = multiprocessing.Process(target=listen_for_messages, args=(self.queue, self.stop_event))
                self.listener = threading.Thread(target=self.listen_for_messages, args=(self.queue, self.stop_event), daemon=True)
                self.threads.append(self.listener)
                self.listener.start()
                #
                # self.listener_info = threading.Thread(target=self.listen_for_info, args=(self.queue, self.stop_event), daemon=True)
                # self.threads.append(self.listener_info)
                # self.listener_info.start()


            else:
                self.mystatusframe.update_status(text=f"{benchmark_name} Benchmark is already running for {gpu_name}. "
                                                      f"Click on the End button to end the benchmarking.")
                print(f"{benchmark_name} Benchmark is already running for {gpu_name}. "
                                                      f"Click on the End button to end the benchmarking.")

        else:
            self.mystatusframe.update_status(text="Select a benchmark to run.")
            print("Select a benchmark to run.")
        return

    def end_benchmark(self):
        benchmark_name = self.mybenchmarkframe.get()

        if self.running == True:
            self.mystatusframe.update_status(text=f"ending {benchmark_name} benchmark")
            print(f"ending {benchmark_name} benchmark")
            self.running = False
            # Call a function to end the benchmark.
            self.stop_event.set()  # Signal the worker to stop
            self.process.join(timeout=10)  # Wait for the worker to exit
            if self.process.is_alive():
                print("Process took too long, terminating.")
                self.process.terminate()
            self.listener.join()

            print("Worker stopped.")
            self.mystatusframe.update_status(text=f"{benchmark_name} benchmark has ended.")
            print(f"{benchmark_name} benchmark has ended.")

            self.queue = multiprocessing.Queue()
            self.stop_event = multiprocessing.Event()
        else:
            self.mystatusframe.update_status(text="No benchmark is running. Select a benchmark to run first.")
            print("No benchmark is running. Select a benchmark to run first.")
        return

    def on_close(self):
        """Stop all threads and processes."""
        print("Stopping all threads and processes...")
        # Stop threads
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=1)  # Allow thread to exit cleanly

        # Terminate processes
        for p in self.processes:
            p.terminate()
            p.join()

        print("All processes and threads stopped.")

class MyTabView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)




        # create tabs
        benchmarker_tab = self.add("Benchmarker")
        benchmarker_tab.grid_columnconfigure((0, 1, 2, 3), weight=1)
        benchmarker_tab.grid_rowconfigure((0, 1, 2), weight=1)

        # add widgets on tabs
        # self.label = ctk.CTkLabel(master=benchmarker_tab, text='hello')
        # self.label.grid(row=0, column=0, padx=20, pady=10)
        #
        # self.button = ctk.CTkButton(benchmarker_tab, command=self.button_click)
        # self.button.grid(row=0, column=1, padx=20, pady=10)

        # self.mygpuframe = MyGPUFrame(master=benchmarker_tab, border_color='white', border_width=5, fg_color='gray')
        # self.mygpuframe.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        #
        # self.mybenchmarkframe = MyBenchmarkFrame(master=benchmarker_tab, border_color="green", border_width=5,
        #                                          fg_color="blue")
        # self.mybenchmarkframe.grid(row=0, column=1, padx=10, columnspan=2, pady=10, sticky="nsew")

        self.myactionframe = MyActionFrame(master=benchmarker_tab, border_color="red", border_width=5,
                                                 fg_color="green")
        self.myactionframe.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        # self.mystatusframe = MyStatusFrame(master=benchmarker_tab, border_color='white', border_width=5, fg_color='orange')
        # self.mystatusframe.grid(row=1, column=0, padx=10, columnspan=3, rowspan=2, pady=10, sticky="nsew")
        #
        # self.mycurrentframe = MyCurrentFrame(master=benchmarker_tab, border_color='white', border_width=5, fg_color='gray')
        # self.mycurrentframe.grid(row=1, column=3, padx=10, rowspan=2, pady=10, sticky="nsew")


        results_tab = self.add("Results")
        results_tab.grid_columnconfigure((0, 1, 2, 3), weight=1)
        results_tab.grid_rowconfigure((0, 1, 2), weight=1)

    def button_click(self):
        print('button is clicked on')
        return

    def on_close(self):
        self.myactionframe.on_close()



# Create App class
class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # Catch close event
        self.title("yero-ml-benchmark")
        self.geometry("1000x750")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)


        # self.checkbox_frame = MyCheckboxFrame(self)
        # self.checkbox_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")

        self.tab_view = MyTabView(master=self, anchor="nw", width=1200, height=750)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    def on_close(self):
        self.tab_view.on_close()

        # Kill any remaining orphaned Python processes
        for proc in multiprocessing.active_children():
            proc.terminate()
            proc.join()

        self.destroy()  # Close the Tkinter window






