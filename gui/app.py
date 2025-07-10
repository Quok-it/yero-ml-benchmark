import multiprocessing
import customtkinter as ctk
from customtkinter.windows.widgets.ctk_tabview import CTkTabview
from gui.action_frame import MyActionFrame
from gui.compare_results_frame import MyResultsCompareFrame
from gui.open_results_frame import MyResultsOpenFrame


ctk.set_appearance_mode("dark")


class App(ctk.windows.ctk_tk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # Catch window X click
        self.title("yero-ml-benchmark")
        self.geometry("1920x1080")
        self.configure(bg="#141414")
        self.update()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tab_view = MyTabView(
            master=self,
            anchor="nw",
            width=self.winfo_width(),
            height=self.winfo_height(),
            fg_color="#141414",
            text_color="#FFB000"
        )
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    def on_close(self):
        self.tab_view.on_close()


        for proc in multiprocessing.active_children():
            proc.terminate()
            proc.join()

        self.destroy()


class MyTabView(CTkTabview):
    def __init__(self, master: App, **kwargs):
        super().__init__(master, **kwargs)

        benchmarker_tab = self.add("Benchmarker")
        benchmarker_tab.grid_columnconfigure((0, 1, 2, 3), weight=1)
        benchmarker_tab.grid_rowconfigure((0, 1, 2), weight=1)

        compare_results_tab = self.add("Compare Results")
        compare_results_tab.grid_columnconfigure((2, 3, 4, 5), weight=1)
        compare_results_tab.grid_rowconfigure((0, 1, 2), weight=1)

        open_results_tab = self.add("Open Results")
        open_results_tab.grid_columnconfigure((2, 3, 4, 5), weight=1)
        open_results_tab.grid_rowconfigure((0, 1, 2), weight=1)

        self.myactionframe = MyActionFrame(
            master=benchmarker_tab,
            border_color="#141414",
            border_width=0,
            fg_color="#141414"
        )
        self.myactionframe.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        self.mycompareresults = MyResultsCompareFrame(
            master=compare_results_tab,
            border_color='#FFB000',
            border_width=3,
            fg_color='#141414'
        )
        self.mycompareresults.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.mycompareresults.benchmarked_gpus(pre_selected=[])

        self.myopenresults = MyResultsOpenFrame(
            master=open_results_tab,
            border_color='#FFB000',
            border_width=3,
            fg_color='#141414'
        )
        self.myopenresults.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def on_close(self):
        self.myactionframe.on_close()
