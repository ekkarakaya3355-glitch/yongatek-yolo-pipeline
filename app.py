from dataclasses import dataclass, field
from typing import Dict

from app.config import Configs
from app.logger import Logger
from app import train,inference

@dataclass
class app:
    """
    Uygulamanın ana sınıfı. Config'e göre ilgili modu çalıştırır.

    Args
    configs(Dict) : Yüklenmiş config sözlüğü.
    """

    configs: Dict
    logger: Logger = field(default=None, init=False)

    def __post_init__(self) -> None:

        self.logger = Logger(**self.configs["logger"])
        self.logger.debug("##########CONFIGURATION##########")
        self.logger.debug(self.configs)

    def run(self) -> None:

        mode = configs.get("mode")

        if mode == "train": 
            train.run(self.configs, self.logger)
        elif mode == "predict":
            inference.run(self.configs, self.logger)
        else:
            self.logger.error(f"Geçersiz mod: '{mode}.")
            raise ValueError(f"Geçersiz mod: '{mode}'. Beklenen 'train' yada 'predict' ")




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Computer Vision Pipeline")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    app(configs=configs).run()
