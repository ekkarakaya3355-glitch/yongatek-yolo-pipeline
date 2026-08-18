from app.config import Configs
from app.logger import Logger
from app import train,inference



def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ [NAME OF PROJECT] CONFIGURATIONS ############")
    logger.debug(configs)

    mode = configs.get("mode")

    if mode == "train": 
        train.run(configs, logger)
    elif mode == "predict":
        inference.run(configs, logger)
    else:
        logger.error(f"Geçersiz mod: '{mode}'. Beklenen 'train' yada 'predict' ")
        raise ValueError(f"Geçersiz mod: '{mode}'")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str)
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    main(args, configs)
