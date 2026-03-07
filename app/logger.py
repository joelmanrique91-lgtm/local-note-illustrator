from __future__ import annotations

import logging
from pathlib import Path

from app.config import AppConfig


class GuiLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.callback(msg)



def setup_logger(config: AppConfig) -> logging.Logger:
    logger = logging.getLogger("local_note_illustrator")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    config.log_dir.mkdir(parents=True, exist_ok=True)
    logfile = Path(config.log_dir) / config.app_log_file

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger



def attach_gui_handler(logger: logging.Logger, callback) -> GuiLogHandler:
    handler = GuiLogHandler(callback)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return handler
