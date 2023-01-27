import yaml
from yaml.loader import SafeLoader

def get_config(config_file: str):
    with  open('config.yaml') as f:
        return yaml.load(f, Loader=SafeLoader)
