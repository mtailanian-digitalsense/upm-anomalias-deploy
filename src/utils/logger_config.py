import logging
import coloredlogs
from collections import deque


class DashHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__()
        self.log_messages = deque(maxlen=capacity)

    def emit(self, record):
        self.log_messages.append(self.format(record))

    def get_log(self):
        return "\n".join(list(self.log_messages)[::-1])

# Formatter without colors for DashHandler
dash_handler = DashHandler()
dash_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
dash_handler.setFormatter(dash_formatter)

# Colored formatter for console handler
console_handler = logging.StreamHandler()
console_formatter = coloredlogs.ColoredFormatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)

logging.basicConfig(level=logging.INFO, handlers=[console_handler, dash_handler])
logger = logging.getLogger(__name__)
