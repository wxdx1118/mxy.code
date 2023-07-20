from Factory_Run import Factory_Run
from webmanager import Webmanager

def main():
    factory=Factory_Run()
    webmanager=Webmanager()
    webmanager.connect(factory)
    
if __name__ == "__main__":
    main()
