from app.config import Configs
from app.logger import Logger


def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ [NAME OF PROJECT] CONFIGURATIONS ############")
    logger.debug(configs)
    print(args)

    print(configs["project"]["name"])




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str)
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    main(args, configs)
