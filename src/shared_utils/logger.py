import logging
import os
from datetime import datetime

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_PATH,
    format="[ %(asctime)s ] %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

# Optional: Add console logging (so logs also show in terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "[ %(asctime)s ] %(levelname)s - %(name)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

logging.getLogger().addHandler(console_handler)