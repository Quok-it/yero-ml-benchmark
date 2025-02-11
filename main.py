
from ui import App



if __name__ == '__main__':


    app = App()
    # Runs the app
    app.mainloop()



    # use_device = "cuda:0"
    #
    # profile_output = "torch_profile.txt"
    # smi_output = "gpu_usage.log"
    # #train_profiler_basic(use_device, profile_output, smi_output)
    #
    # bm(use_device)
