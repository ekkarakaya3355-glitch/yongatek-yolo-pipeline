from dataclasses import dataclass
from typing import Dict
import os
import toml

@dataclass
class Configs:

    configs_folder_path:str="./configs"

    def load(self, config_name:str)->Dict:
        if os.environ.get("CONFIG_FILE") is not None:
            config_name = os.environ["CONFIG_FILE"]
        config_file_path = os.path.join(self.configs_folder_path, f"config.{config_name}.toml")
        assert os.path.isfile(config_file_path), f"Given configuration file is not exist. File: {config_file_path}"
        configs = toml.load(config_file_path)

        return configs

if __name__ == "__main__":
    pass