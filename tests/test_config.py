import unittest
import os

from app.config import Configs

class TestConfigs(unittest.TestCase):

    def test_load_environments(self,):
        configs = Configs()
        config_file_paths = os.listdir(configs.configs_folder_path)
        for config in config_file_paths:
            config_name = config.replace("config","").replace(".","").replace("toml","")
            configs.load(config_name)


if __name__ == '__main__':
    unittest.main()
