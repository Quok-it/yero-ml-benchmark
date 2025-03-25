import os
from gui.app import App
import faulthandler
faulthandler.enable()

if __name__ == '__main__':
    app = App()
    app.mainloop()