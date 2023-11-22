import logging


logging.basicConfig(
        level=logging.INFO,
        filename="file_logs.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s"
        )