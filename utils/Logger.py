import os

def log(msg):
    with open(os.path.join("logs", "log.log"), "a") as f:
        f.write(msg + "\n")