import yaml

def load_config(file_path):
    with open('config.yaml','r') as read_data:
        return yaml.safe_load(read_data)
