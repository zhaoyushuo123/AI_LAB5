import logging
import os
from config import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

if not os.path.exists('./log/'):
    os.mkdir('./log/')
fh = logging.FileHandler('./log/log-' + config["dataset"] + '-' + config["model_name"] + '.log', mode='a',
                         encoding='utf-8')
fh.setFormatter(formatter)

console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
console.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(console)