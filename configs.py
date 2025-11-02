import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/04_sentence_recognition", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 64 # original 96
        self.width = 1024 #original 1408
        self.max_text_length = 0
        self.batch_size = 64 # Original = 32
        self.learning_rate = 0.0005
        self.train_epochs = 10 # Original = 1000
        self.train_workers = 4 # Original = 20