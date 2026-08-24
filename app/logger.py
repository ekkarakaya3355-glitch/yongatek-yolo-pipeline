from dataclasses import dataclass
from loguru import logger as builtin_logger


@dataclass
class Logger:
    
    filepath:str
    rotation:str

    def __post_init__(self,)->None:
        self.logger = builtin_logger
        self.logger.remove(0)
        msg_format = "[{time:YYYY-MM-DD HH:mm:ss}] | {level} | [{module}:{function}:{line}] | [{message}]"
        self.logger.add(sink=self.filepath, rotation=self.rotation, encoding="utf-8", format=msg_format)


    def debug(self,message)->None:
        self.logger.opt(depth=1).debug(message)


    def info(self,message)->None:
        self.logger.opt(depth=1).info(message)


    def warning(self, message)->None:
        self.logger.opt(depth=1).warning(message)


    def error(self,message)->None:
        self.logger.opt(depth=1).error(message)


if __name__ == "__main__":
    pass
