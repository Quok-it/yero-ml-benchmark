from cli.cli import CLI
import faulthandler
faulthandler.enable()

if __name__ == '__main__':
    app = CLI()
    app.main()
