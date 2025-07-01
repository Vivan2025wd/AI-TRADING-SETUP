# utils/logger.py

import logging
import os

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'storage', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Logger config
logger = logging.getLogger("trading_system")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Shortcut function
log = logger