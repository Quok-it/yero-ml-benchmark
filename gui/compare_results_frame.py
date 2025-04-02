import customtkinter as ctk
import tkinter as tk
from pathlib import Path
import ijson
from data_processing.data_analysis import create_table_plots

font_type = {'family': 'Consolas', 'size': 14}

class MyResultsCompareFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.gpu_shown = None
        self.gpus_available = None
        self.benchmarked_dir = None
        self.selected_gpus = None
        self.pre_selected = None
        self.variable = ctk.StringVar()

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.gpu_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                      fg_color="#141414")

        self.myresultsscriptframe = MyResultsScriptFrame(master=master,
                                                         border_color='#FFB000',
                                                         border_width=3,
                                                         fg_color='#141414')
        self.myresultsscriptframe.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


    def benchmarked_gpus(self, pre_selected):
        self.pre_selected = pre_selected
        self.gpus_available = []
        self.gpu_shown = []

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.gpu_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame, fg_color="#141414")
        device_id = ''

        current_path = Path(__file__).resolve()
        benchmarked_dir = current_path.parent.parent / 'benchmarked'
        benchmarked_dir.mkdir(exist_ok=True)

        self.benchmarked_dir = benchmarked_dir

        '''
        This loop looks for valid json trace files and gets the GPU name and device id to display on the GUI
        '''
        for gpu_folder in benchmarked_dir.iterdir():
            if gpu_folder.is_dir():
                for script_folder in gpu_folder.iterdir():
                    if script_folder.is_dir():
                        for model_folder in script_folder.iterdir():
                            if model_folder.is_dir():
                                for params_folder in model_folder.iterdir():
                                    if params_folder.is_dir():
                                        for trace_file in params_folder.iterdir():
                                            if trace_file.is_file():
                                                with open(trace_file, 'r') as f:
                                                    parser = ijson.parse(f)
                                                    device_id = None

                                                    try:
                                                        for prefix, event, value in parser:
                                                            if prefix == 'benchmark_info.device_id':
                                                                device_id = value
                                                                break
                                                    except Exception as e:
                                                        continue

                                                    if device_id is None:
                                                        continue

                                                self.gpus_available.append([gpu_folder, device_id])
                                                device_id = ''

        unique_tuples = {tuple(sublist) for sublist in self.gpus_available}
        self.gpus_available = [list(tup) for tup in unique_tuples]

        self.checkbox_gpu_refresh = ctk.CTkCheckBox(self.gpu_frame,
                                                    text='Refresh the GPU List',
                                                    onvalue='Refreshing',
                                                    offvalue='ignore',
                                                    variable=self.variable,
                                                    command=self.restart_start_results,
                                                    text_color="#FFB000",
                                                    font=(font_type['family'], font_type['size']))
        self.checkbox_gpu_refresh.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if self.gpus_available == []:
            self.label_gpu_refresh = ctk.CTkLabel(self.gpu_frame,
                                                  text='The button(s) (if) selected previously\n'
                                                       'have no common files.\n\n'
                                                       'Please upload the folders/files properly\n'
                                                       'and press the Refresh button above.',
                                                  text_color="#FFB000",
                                                  font=(font_type['family'], font_type['size']))
            self.label_gpu_refresh.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        for index, gpu in enumerate(self.gpus_available):
            var = ctk.StringVar()
            checkbox_gpu = ctk.CTkCheckBox(self.gpu_frame,
                                           text=str(gpu[0]).split('/')[-1] + '\n' + gpu[1],
                                           onvalue=str(gpu[0]) + ',' +gpu[1],
                                           offvalue='ignore',
                                           variable=var,
                                           command=self.get_gpus_selected,
                                           text_color="#FFB000",
                                           font=(font_type['family'], font_type['size']))
            checkbox_gpu.grid(row=index + 1, column=0, padx=10, pady=10, sticky="nsew")

            checkbox_gpu.select()
            if self.pre_selected is not None:
                if checkbox_gpu.get() not in self.pre_selected:
                    checkbox_gpu.deselect()

            self.gpu_shown.append(checkbox_gpu)

        self.scrollable_canvas.add_content(self.gpu_frame, row=0, column=0)

    def get_gpus_selected(self):
        self.myresultsscriptframe.myresultsmodelframe.clear_frame()
        self.selected_gpus = [checkbox.get() for checkbox in self.gpu_shown if checkbox.get() != 'ignore']
        self.myresultsscriptframe.benchmarked_scripts(self.benchmarked_dir,
                                                      self.selected_gpus,
                                                      pre_selected=None)

    def restart_start_results(self):
        if self.gpu_shown is not None:
            self.pre_selected = [checkbox.get() for checkbox in self.gpu_shown if checkbox.get() != 'ignore']
            if self.checkbox_gpu_refresh is not None:
                self.checkbox_gpu_refresh.deselect()
        self.clear_frame()
        self.benchmarked_gpus(self.pre_selected)
        self.myresultsscriptframe.restart_start_results()

    def clear_frame(self):
        self.myresultsscriptframe.myresultsmodelframe.clear_frame()
        for widget in self.winfo_children():
            widget.destroy()

class MyResultsScriptFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.selected_scripts = None
        self.pre_selected = None
        self.label_script_refresh = None
        self.radio_script_refresh = None
        self.models_shown = None
        self.models_available = None
        self.benchmarked_dir = None
        self.selected_gpus = None
        self.selected_model = None
        self.scripts_available = []
        self.scripts_shown = []

        self.variable = ctk.StringVar()

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.myresultsmodelframe = MyResultsModelFrame(master=master,
                                                       border_color='#FFB000',
                                                       border_width=3,
                                                       fg_color='#141414')
        self.myresultsmodelframe.grid(row=1, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    def benchmarked_scripts(self, benchmarked_dir, selected_gpus, pre_selected):
        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.pre_selected = pre_selected
        self.benchmarked_dir = benchmarked_dir
        self.selected_gpus = selected_gpus
        self.scripts_available = []
        self.scripts_shown = []

        if self.selected_gpus is None:
            return

        gpu_names = [g_n.split(',')[0].split('/')[-1] for g_n in self.selected_gpus]
        device_ids = [g_n.split(',')[1] for g_n in self.selected_gpus]

        '''
        This loop looks through the GPU(s) selected previously 
        and finds shared script(s) between the GPU(s) selected.
        It confirms it by the valid json trace files.
        '''
        selected_benchmark_path = Path(benchmarked_dir)
        for gpu_folder in selected_benchmark_path.iterdir():
            g_name = str(gpu_folder).split('/')[-1]
            if g_name in gpu_names:
                gpu_id_needed = [g_n.split(',')[1]  for g_n in self.selected_gpus if g_name == g_n.split(',')[0].split('/')[-1]]
                shared_scripts = []
                for script_folder in gpu_folder.iterdir():
                    if script_folder.is_dir():
                        for model_folder in script_folder.iterdir():
                            if model_folder.is_dir():
                                for params_folder in model_folder.iterdir():
                                    if params_folder.is_dir():
                                        check_files = []
                                        for trace_file in params_folder.iterdir():
                                            if trace_file.is_file():
                                                with open(trace_file, 'r') as f:
                                                    parser = ijson.parse(f)
                                                    device_id = None

                                                    try:
                                                        for prefix, event, value in parser:
                                                            if prefix == 'benchmark_info.device_id':
                                                                device_id = value
                                                                if device_id in gpu_id_needed:
                                                                    check_files.append(device_id)
                                                                break
                                                    except Exception as e:
                                                        continue

                                                    if device_id is None:
                                                        continue

                                        check_files = list(set(check_files))
                                        if check_files == []:
                                            continue

                                        if sorted(gpu_id_needed) == sorted(check_files):
                                            shared_scripts.append(str(script_folder).split('/')[-1])

                if shared_scripts != [] and self.scripts_available == []:
                    self.scripts_available = list(set(shared_scripts))

                elif shared_scripts != []:
                    self.scripts_available = list(set(self.scripts_available) & set(shared_scripts))

        self.scripts_available.sort()

        self.radio_script_refresh = ctk.CTkCheckBox(self.radio_frame,
                                                    text='Refresh the Script List',
                                                    onvalue='Refreshing',
                                                    offvalue='ignore',
                                                    variable=self.variable,
                                                    command=self.restart_start_results,
                                                    text_color="#FFB000",
                                                    font=(font_type['family'], font_type['size']))
        self.radio_script_refresh.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if self.scripts_available == []:
            self.label_script_refresh = ctk.CTkLabel(self.radio_frame,
                                                     text='The button(s) (if) selected previously\n'
                                                          'have no common files.\n\n'
                                                          'Please upload the folders/files properly\n'
                                                          'and press the Refresh button above.',
                                                     text_color="#FFB000",
                                                     font=(font_type['family'], font_type['size']))
            self.label_script_refresh.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        for index, script in enumerate(self.scripts_available):
            var = ctk.StringVar()
            checkbox_script = ctk.CTkCheckBox(self.radio_frame,
                                              text=script,
                                              onvalue=script,
                                              offvalue='ignore',
                                              variable=var,
                                              command=self.get_scripts_selected,
                                              text_color="#FFB000",
                                              font=(font_type['family'], font_type['size']))
            checkbox_script.grid(row=index + 1, column=0, padx=10, pady=10, sticky="nsew")

            if self.pre_selected is not None:
                if checkbox_script.get() not in self.pre_selected:
                    checkbox_script.deselect()

            self.scripts_shown.append(checkbox_script)

        self.scrollable_canvas.add_content(self.radio_frame, row=0, column=0)

        return

    def get_scripts_selected(self):
        self.myresultsmodelframe.clear_frame()
        self.selected_scripts = [checkbox.get() for checkbox in self.scripts_shown if checkbox.get() != 'ignore']
        self.myresultsmodelframe.benchmarked_models(self.benchmarked_dir,
                                                    self.selected_gpus,
                                                    self.selected_scripts,
                                                    pre_selected=None)

    def restart_start_results(self):
        if self.scripts_shown is not None:
            self.pre_selected = [checkbox.get() for checkbox in self.scripts_shown if checkbox.get() != 'ignore']
            if self.radio_script_refresh is not None:
                self.radio_script_refresh.deselect()
        self.clear_frame()
        self.benchmarked_scripts(self.benchmarked_dir,
                                 self.selected_gpus,
                                 self.pre_selected)


    def clear_frame(self):
        self.myresultsmodelframe.clear_frame()
        for widget in self.winfo_children():
            widget.destroy()

class MyResultsModelFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.selected_models = None
        self.pre_selected = None
        self.label_model_refresh = None
        self.radio_model_refresh = None
        self.models_shown = None
        self.models_available = None
        self.benchmarked_dir = None
        self.selected_gpus = None
        self.selected_model = None
        self.selected_script = None

        self.variable = ctk.StringVar()

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.myresultsparamsframe = MyResultsParamsFrame(master=master,
                                                         border_color='#FFB000',
                                                         border_width=3,
                                                         fg_color='#141414')
        self.myresultsparamsframe.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    def benchmarked_models(self, benchmarked_dir, selected_gpus, selected_script, pre_selected):
        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.benchmarked_dir = benchmarked_dir
        self.selected_gpus = selected_gpus
        self.selected_script = selected_script
        self.pre_selected = pre_selected
        self.models_available = []
        self.models_shown = []
        gpu_names = [g_n.split(',')[0].split('/')[-1]  for g_n in selected_gpus]
        device_ids = [g_n.split(',')[1]for g_n in selected_gpus]

        '''
        This loop looks through the GPU(s) and script(s) selected previously 
        and finds shared models between the script(s) selected.
        It confirms it by the valid json trace files.
        '''
        selected_benchmark_path = Path(benchmarked_dir)
        for gpu_folder in selected_benchmark_path.iterdir():
            g_name = str(gpu_folder).split('/')[-1]
            if g_name in gpu_names:
                gpu_id_needed = [g_n.split(',')[1] for g_n in self.selected_gpus if g_name == g_n.split(',')[0].split('/')[-1]]
                for script_folder in gpu_folder.iterdir():
                    if script_folder.is_dir():
                        s_name = str(script_folder).split('/')[-1]
                        if s_name in self.selected_script:
                            shared_models = []
                            for model_folder in script_folder.iterdir():
                                if model_folder.is_dir():
                                    for params_folder in model_folder.iterdir():
                                        if params_folder.is_dir():
                                            check_files = []
                                            for trace_file in params_folder.iterdir():
                                                if trace_file.is_file():
                                                    with open(trace_file, 'r') as f:
                                                        parser = ijson.parse(f)
                                                        device_id = None

                                                        try:
                                                            for prefix, event, value in parser:
                                                                if prefix == 'benchmark_info.device_id':
                                                                    device_id = value
                                                                    if device_id in gpu_id_needed:
                                                                        check_files.append(device_id)
                                                                    break
                                                        except Exception as e:
                                                            continue

                                                        if device_id is None:
                                                            continue

                                            check_files = list(set(check_files))
                                            if check_files == []:
                                                continue

                                            if sorted(gpu_id_needed) == sorted(check_files):
                                                shared_models.append(str(model_folder).split('/')[-1])

                            if shared_models != [] and self.models_available == []:
                                self.models_available = list(set(shared_models))

                            elif shared_models != []:
                                self.models_available = list(set(self.models_available) & set(shared_models))

        self.models_available.sort()

        self.radio_model_refresh = ctk.CTkCheckBox(self.radio_frame,
                                                   text='Refresh the Model List',
                                                   onvalue='Refreshing',
                                                   offvalue='ignore',
                                                   variable=self.variable,
                                                   command=self.restart_start_results,
                                                   text_color="#FFB000",
                                                   font=(font_type['family'], font_type['size']))
        self.radio_model_refresh.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if self.models_available == []:
            self.label_model_refresh = ctk.CTkLabel(self.radio_frame,
                                                    text='The button(s) (if) selected previously\n'
                                                          'have no common files.\n\n'
                                                          'Please upload the folders/files properly\n'
                                                          'and press the Refresh button above.',
                                                    text_color="#FFB000",
                                                    font=(font_type['family'], font_type['size']))
            self.label_model_refresh.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        for index, model in enumerate(self.models_available):
            var = ctk.StringVar()
            checkbox_model = ctk.CTkCheckBox(self.radio_frame,
                                             text=model.split('.json')[0],
                                             onvalue=model,
                                             offvalue='ignore',
                                             variable=var,
                                             command=self.get_params_selected,
                                             text_color="#FFB000",
                                             font=(font_type['family'], font_type['size']))
            checkbox_model.grid(row=index + 1, column=0, padx=10, pady=10, sticky="nsew")

            if self.pre_selected is not None:
                if checkbox_model.get() not in self.pre_selected:
                    checkbox_model.deselect()

            self.models_shown.append(checkbox_model)

        self.scrollable_canvas.add_content(self.radio_frame, row=0, column=0)
        return

    def get_params_selected(self):
        self.myresultsparamsframe.clear_frame()
        self.selected_models = [checkbox.get() for checkbox in self.models_shown if checkbox.get() != 'ignore']
        self.myresultsparamsframe.benchmarked_params(self.benchmarked_dir,
                                                     self.selected_gpus,
                                                     self.selected_script,
                                                     self.selected_models,
                                                     pre_selected=None)

    def restart_start_results(self):
        if self.models_shown is not None:
            self.pre_selected = [checkbox.get() for checkbox in self.models_shown if checkbox.get() != 'ignore']
            if self.radio_model_refresh is not None:
                self.radio_model_refresh.deselect()
        self.clear_frame()
        self.benchmarked_models(self.benchmarked_dir,
                                self.selected_gpus,
                                self.selected_script,
                                self.pre_selected)

    def clear_frame(self):
        self.myresultsparamsframe.clear_frame()
        for widget in self.winfo_children():
            widget.destroy()

class MyResultsParamsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.pre_selected = None
        self.label_params_refresh = None
        self.radio_params_refresh = None
        self.params_shown = None
        self.params_available = None
        self.benchmarked_dir = None
        self.selected_gpus = None
        self.selected_model = None
        self.selected_script = None
        self.selected_params = None

        self.variable = ctk.StringVar()

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.myresultsexecuteframe = MyResultsExecuteFrame(master=master,
                                                           border_color='#FFB000',
                                                           border_width=3,
                                                           fg_color='#141414')
        self.myresultsexecuteframe.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")


    def benchmarked_params(self, benchmarked_dir, selected_gpus, selected_script, selected_model, pre_selected):
        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.benchmarked_dir = benchmarked_dir
        self.selected_gpus = selected_gpus
        self.selected_script = selected_script
        self.selected_model = selected_model
        self.pre_selected = pre_selected
        self.params_available = []
        self.params_shown = []
        gpu_names = [g_n.split(',')[0].split('/')[-1]  for g_n in selected_gpus]
        device_ids = [g_n.split(',')[1]for g_n in selected_gpus]

        '''
        This loop looks through the GPU(s), script(s), model(s) selected previously 
        and finds shared models between the model(s) selected.
        It confirms it by the valid json trace files.
        '''
        selected_benchmark_path = Path(benchmarked_dir)
        for gpu_folder in selected_benchmark_path.iterdir():
            g_name = str(gpu_folder).split('/')[-1]
            if g_name in gpu_names:
                gpu_id_needed = [g_n.split(',')[1] for g_n in self.selected_gpus if
                                 g_name == g_n.split(',')[0].split('/')[-1]]
                for script_folder in gpu_folder.iterdir():
                    if script_folder.is_dir():
                        s_name = str(script_folder).split('/')[-1]
                        if s_name in self.selected_script:
                            for model_folder in script_folder.iterdir():
                                if model_folder.is_dir():
                                    m_name = str(model_folder).split('/')[-1]
                                    if m_name in self.selected_model:
                                        shared_params= []
                                        for params_folder in model_folder.iterdir():
                                            if params_folder.is_dir():
                                                check_files = []
                                                for trace_file in params_folder.iterdir():
                                                    if trace_file.is_file():
                                                        with open(trace_file, 'r') as f:
                                                            parser = ijson.parse(f)
                                                            device_id = None

                                                            try:
                                                                for prefix, event, value in parser:
                                                                    if prefix == 'benchmark_info.device_id':
                                                                        device_id = value
                                                                        if device_id in gpu_id_needed:
                                                                            check_files.append(device_id)
                                                                        break
                                                            except Exception as e:
                                                                continue

                                                            if device_id is None:
                                                                continue

                                                check_files = list(set(check_files))
                                                if check_files == []:
                                                    continue

                                                if sorted(gpu_id_needed) == sorted(check_files):
                                                    shared_params.append(str(params_folder).split('/')[-1])

                                        if shared_params != [] and self.params_available == []:
                                            self.params_available = list(set(shared_params))

                                        elif shared_params != []:
                                            self.params_available = list(set(self.params_available) & set(shared_params))

        self.params_available = list(set(self.params_available))
        self.params_available.sort()

        self.radio_params_refresh = ctk.CTkCheckBox(self.radio_frame,
                                                    text='Refresh the Params List',
                                                    onvalue='Refreshing',
                                                    offvalue='ignore',
                                                    variable=self.variable,
                                                    command=self.restart_start_results,
                                                    text_color="#FFB000",
                                                    font=(font_type['family'], font_type['size']))
        self.radio_params_refresh.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if self.params_available == []:
            self.label_params_refresh = ctk.CTkLabel(self.radio_frame,
                                                     text='The button(s) (if) selected previously\n'
                                                          'have no common files.\n\n'
                                                          'Please upload the folders/files properly\n'
                                                          'and press the Refresh button above.',
                                                     text_color="#FFB000",
                                                     font=(font_type['family'], font_type['size']))
            self.label_params_refresh.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        for index, param in enumerate(self.params_available):
            var = ctk.StringVar()
            checkbox_params = ctk.CTkCheckBox(self.radio_frame,
                                              text=param,
                                              onvalue=param,
                                              offvalue='ignore',
                                              variable=var,
                                              command=self.get_execution_selected,
                                              text_color="#FFB000",
                                              font=(font_type['family'], font_type['size']))
            checkbox_params.grid(row=index + 1, column=0, padx=10, pady=10, sticky="nsew")

            if self.pre_selected is not None:
                if checkbox_params.get() not in self.pre_selected:
                    checkbox_params.deselect()

            self.params_shown.append(checkbox_params)

        self.scrollable_canvas.add_content(self.radio_frame, row=0, column=0)

        return

    def get_execution_selected(self):
        self.myresultsexecuteframe.clear_frame()
        self.selected_params = [checkbox.get() for checkbox in self.params_shown if checkbox.get() != 'ignore']
        self.myresultsexecuteframe.benchmarked_execute(self.benchmarked_dir,
                                                       self.selected_gpus,
                                                       self.selected_script,
                                                       self.selected_model,
                                                       self.selected_params)

    def restart_start_results(self):
        if self.params_shown is not None:
            self.pre_selected = [checkbox.get() for checkbox in self.params_shown if checkbox.get() != 'ignore']
            if self.radio_params_refresh is not None:
                self.radio_params_refresh.deselect()
        self.clear_frame()
        self.benchmarked_params(self.benchmarked_dir,
                                self.selected_gpus,
                                self.selected_script,
                                self.selected_model,
                                self.pre_selected)

    def clear_frame(self):
        self.myresultsexecuteframe.clear_frame()
        for widget in self.winfo_children():
            widget.destroy()

class MyResultsExecuteFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.radio_execute_run = None
        self.selected_executes = None
        self.pre_selected = None
        self.label_execute_refresh = None
        self.radio_execute_refresh = None
        self.execute_shown = None
        self.execute_available = None
        self.benchmarked_dir = None
        self.selected_gpus = None
        self.selected_model = None
        self.selected_script = None
        self.selected_params = None
        self.selected_execute = None

        self.variable = ctk.StringVar()

        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.myresultschartframe = MyResultsChartFrame(master=master,
                                                       border_color='#FFB000',
                                                       border_width=3,
                                                       fg_color='#141414')
        self.myresultschartframe.grid(row=0, column=2, padx=10, rowspan=3, columnspan=4, pady=10, sticky="nsew")


    def benchmarked_execute(self, benchmarked_dir, selected_gpus, selected_script, selected_model, selected_params, pre_selected=None):
        self.scrollable_canvas = ScrollableCanvas(self)
        self.scrollable_canvas.grid(row=0, column=0, sticky="nsew")

        self.radio_frame = ctk.CTkFrame(self.scrollable_canvas.content_frame,
                                        fg_color="#141414")

        self.benchmarked_dir = benchmarked_dir
        self.selected_gpus = selected_gpus
        self.selected_script = selected_script
        self.selected_model = selected_model
        self.selected_params = selected_params
        self.pre_selected = pre_selected

        self.execute_available = []
        self.execute_shown = []

        gpu_names = [g_n.split(',')[0].split('/')[-1]  for g_n in selected_gpus]
        device_ids = [g_n.split(',')[1]for g_n in selected_gpus]

        '''
        This loop looks through the GPU(s), script(s), model(s), param(s) selected previously 
        and finds shared models between the param(s) selected.
        It confirms it by the valid json trace files.
        '''
        selected_benchmark_path = Path(benchmarked_dir)
        for gpu_folder in selected_benchmark_path.iterdir():
            g_name = str(gpu_folder).split('/')[-1]
            if g_name in gpu_names:
                gpu_id_needed = [g_n.split(',')[1] for g_n in self.selected_gpus if g_name == g_n.split(',')[0].split('/')[-1]]
                for script_folder in gpu_folder.iterdir():
                    if script_folder.is_dir():
                        s_name = str(script_folder).split('/')[-1]
                        if s_name in self.selected_script:
                            for model_folder in script_folder.iterdir():
                                if model_folder.is_dir():
                                    m_name = str(model_folder).split('/')[-1]
                                    if m_name in self.selected_model:
                                        for params_folder in model_folder.iterdir():
                                            if params_folder.is_dir():
                                                p_name = str(params_folder).split('/')[-1]
                                                if p_name in self.selected_params:
                                                    shared_execute = []
                                                    check_files = []
                                                    for trace_file in params_folder.iterdir():
                                                        if trace_file.is_file():
                                                            with open(trace_file, 'r') as f:
                                                                parser = ijson.parse(f)
                                                                device_id = None

                                                                try:
                                                                    for prefix, event, value in parser:
                                                                        if prefix == 'benchmark_info.device_id':
                                                                            device_id = value
                                                                            if device_id in gpu_id_needed:
                                                                                check_files.append(device_id)
                                                                            break
                                                                except Exception as e:
                                                                    continue

                                                                if device_id is None:
                                                                    continue

                                                            check_files = list(set(check_files))
                                                            if check_files == []:
                                                                continue

                                                            if sorted(gpu_id_needed) == sorted(check_files):
                                                                shared_execute.append(str(trace_file)
                                                                                      .split('/')[-1]
                                                                                      .split('.')[0]
                                                                                      .split('_')[-1])

                                                    if shared_execute != [] and self.execute_available == []:
                                                        self.execute_available = list(set(shared_execute))

                                                    elif shared_execute != []:
                                                        self.execute_available = list(set(self.execute_available) & set(shared_execute))

        self.execute_available.sort()

        self.radio_execute_refresh = ctk.CTkCheckBox(self.radio_frame,
                                                     text='Refresh the Execute List',
                                                     onvalue='Refreshing',
                                                     offvalue='ignore',
                                                     variable=self.variable,
                                                     command=self.restart_start_results,
                                                     text_color="#FFB000",
                                                     font=(font_type['family'], font_type['size']))
        self.radio_execute_refresh.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if self.execute_available == []:
            self.label_execute_refresh = ctk.CTkLabel(self.radio_frame,
                                                      text='The button(s) (if) selected previously\n'
                                                           'have no common files.\n\n'
                                                           'Please upload the folders/files properly\n'
                                                           'and press the Refresh button above.',
                                                      text_color="#FFB000",
                                                      font=(font_type['family'], font_type['size']))
            self.label_execute_refresh.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        last_button_run_count = 0
        for index, execute in enumerate(self.execute_available):
            var = ctk.StringVar()
            checkbox_execute = ctk.CTkCheckBox(self.radio_frame,
                                               text=execute.split('.json')[0],
                                               onvalue=execute,
                                               offvalue='ignore',
                                               variable=var,
                                               text_color="#FFB000",
                                               font=(font_type['family'], font_type['size']))
            checkbox_execute.grid(row=index + 1, column=0, padx=10, pady=10, sticky="nsew")
            last_button_run_count = index + 2

            if self.pre_selected is not None:
                if checkbox_execute.get() not in self.pre_selected:
                    checkbox_execute.deselect()

            self.execute_shown.append(checkbox_execute)

        self.radio_execute_run = ctk.CTkRadioButton(self.radio_frame,
                                                    text='Click to create the table and plots.',
                                                    value='run',
                                                    variable=self.variable,
                                                    command=self.get_chart_selected,
                                                    text_color="#FFB000",
                                                    font=(font_type['family'], font_type['size']))
        self.radio_execute_run.grid(row=last_button_run_count, column=0, padx=10, pady=10, sticky="nsew")
        self.radio_execute_run.deselect()

        self.scrollable_canvas.add_content(self.radio_frame, row=0, column=0)
        return

    def get_chart_selected(self):
        self.myresultschartframe.clear_frame()
        self.selected_executes = [checkbox.get() for checkbox in self.execute_shown if checkbox.get() != 'ignore']
        self.radio_execute_run.deselect()
        self.myresultschartframe.benchmarked_charts(self.benchmarked_dir,
                                                    self.selected_gpus,
                                                    self.selected_script,
                                                    self.selected_model,
                                                    self.selected_params,
                                                    self.selected_executes)

    def restart_start_results(self):
        if self.execute_shown is not None:
            self.pre_selected = [checkbox.get() for checkbox in self.execute_shown if checkbox.get() != 'ignore']
            if self.radio_execute_refresh is not None:
                self.radio_execute_refresh.deselect()
        self.clear_frame()
        self.benchmarked_execute(self.benchmarked_dir,
                                 self.selected_gpus,
                                 self.selected_script,
                                 self.selected_model,
                                 self.selected_params,
                                 self.pre_selected)

    def clear_frame(self):
        self.myresultschartframe.clear_frame()
        for widget in self.winfo_children():
            widget.destroy()

class MyResultsChartFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.full_table = None

    def benchmarked_charts(self, benchmarked_dir, selected_gpus, selected_script, selected_model, selected_params, selected_execution):
        charts_available = []

        gpu_names = [g_n.split(',')[0].split('/')[-1] for g_n in selected_gpus]
        device_ids = [g_n.split(',')[1] for g_n in selected_gpus]

        '''
        This loop looks through the GPU(s), script(s), model(s), param(s), and execute(s) selected previously 
        and finds shared models between the execute(s) selected.
        It confirms it by the valid json trace files.
        '''
        selected_benchmark_path = Path(benchmarked_dir)
        for gpu_folder in selected_benchmark_path.iterdir():
            g_name = str(gpu_folder).split('/')[-1]
            if g_name in gpu_names:
                gpu_id_needed = [g_n.split(',')[1] for g_n in selected_gpus if g_name == g_n.split(',')[0].split('/')[-1]]
                for script_folder in gpu_folder.iterdir():
                    if script_folder.is_dir():
                        s_name = str(script_folder).split('/')[-1]
                        if s_name in selected_script:
                            for model_folder in script_folder.iterdir():
                                if model_folder.is_dir():
                                    m_name = str(model_folder).split('/')[-1]
                                    if m_name in selected_model:
                                        for params_folder in model_folder.iterdir():
                                            if params_folder.is_dir():
                                                p_name = str(params_folder).split('/')[-1]
                                                if p_name in selected_params:
                                                    for trace_file in params_folder.iterdir():
                                                        if trace_file.is_file():
                                                            with open(trace_file, 'r') as f:
                                                                parser = ijson.parse(f)
                                                                device_id = None
                                                                try:
                                                                    for prefix, event, value in parser:
                                                                        if prefix == 'benchmark_info.device_id':
                                                                            device_id = value
                                                                            if device_id in gpu_id_needed:
                                                                                e_name = str(trace_file).split('/')[-1].split('.')[0].split('_')[-1]
                                                                                e_full_name = str(trace_file)
                                                                                if e_name in selected_execution:
                                                                                    charts_available.append(e_full_name)
                                                                            break
                                                                except Exception as e:
                                                                    continue

                                                                if device_id is None:
                                                                    continue

        table_string, graph_dir = create_table_plots(charts_available, 'results_compare')
        graph_dir = 'yero-ml-benchmark/benchmark_results' + graph_dir.split('benchmark_results')[1]
        table_string = f'Graphs have been generated based on the tables below,\nthey are stored at: {graph_dir}\n\n\n' + table_string

        self.full_table = ctk.CTkTextbox(master=self,
                                         height=300,
                                         width=150,
                                         font=(font_type['family'], font_type['size']),
                                         text_color="#FFB000",
                                         fg_color="#141414")
        self.full_table.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.full_table.insert("0.0", table_string)
        self.full_table.configure(state="disabled", wrap="none")

        return

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

class ScrollableCanvas(ctk.CTkFrame):
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

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

        self.canvas.create_window((0,0),
                                  window=self.content_frame,
                                  anchor="nw")
        self.min_width = -1
        self.min_height = -1

        self.canvas.bind("<Configure>", self.on_resize)


    def on_resize(self, event):
        if self.min_width != -1:
            max_width = max(self.canvas.winfo_width(), self.min_width)
            max_height = max(self.canvas.winfo_height(), self.min_height)

            self.canvas.create_window((0, 0),
                                      window=self.content_frame,
                                      anchor="nw",
                                      height=max_height,
                                      width=max_width)


    def add_content(self, content, row, column, rowspan=1, columnspan=1, padx=10, pady=10, sticky="nsew"):
        content.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky)

        self.content_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.update_idletasks()

        self.min_width = max(self.canvas.winfo_width(), self.content_frame.winfo_width())
        self.min_height = max(self.canvas.winfo_height(), self.content_frame.winfo_height())
