from app.config import load_config
from app.gui import launch_gui
from app.logger import setup_logger



def main() -> None:
    config = load_config()
    logger = setup_logger(config)
    logger.info("Iniciando Local Note Illustrator")
    launch_gui(config, logger)


if __name__ == "__main__":
    main()
