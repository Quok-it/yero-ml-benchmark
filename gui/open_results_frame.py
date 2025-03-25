import customtkinter as ctk
from tkinter import filedialog as fd

font_type = {'family': 'Consolas', 'size': 14}

class MyResultsOpenFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        open_button = ctk.CTkButton(self,
                                    text="Open Table Result",
                                    command=self.open_results,
                                    text_color="#FFB000",
                                    font=(font_type['family'], font_type['size']))

        open_button.grid(column=0, row=0, sticky='w', padx=10, pady=10)

        self.myresultsopenchart = MyResultsChartFrame(master=master,
                                                      border_color='#FFB000',
                                                      border_width=3,
                                                      fg_color='#141414')
        self.myresultsopenchart.grid(row=0, column=1, rowspan=3, columnspan=5, padx=10, pady=10, sticky="nsew")

    def open_results(self):
        filetypes = (
            ('text files', '*.txt'),
        )
        f = fd.askopenfile(filetypes=filetypes)
        if f:
            content = f.read()
            f.close()
            self.myresultsopenchart.show_results(content)

class MyResultsChartFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.full_table = None

    def show_results(self, table_string):
        self.full_table = ctk.CTkTextbox(master=self,
                                         height=300,
                                         width=150,
                                         font=(font_type['family'], font_type['size']),
                                         text_color="#FFB000",
                                         fg_color="#141414")
        self.full_table.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.full_table.insert("0.0", table_string)
        self.full_table.configure(state="disabled", wrap="none")
