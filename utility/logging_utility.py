import os
import numpy as np

import seaborn as sns
from tensorboardX import SummaryWriter

class logwrapper(object):
    def __init__(self, log_path):
        self.writter = SummaryWriter(log_path)
    
    def add_scalar(self, name, scalar, epoch):
        self.writter.add_scalar(name, scalar, epoch)
        self.writter.flush()

    def add_scalars(self, name, scalars, epoch):
        self.writter.add_scalars(name, scalars, epoch)
        self.writter.flush()

    def add_text(self, title, content):
        self.writter.add_text(title, content)
        self.writter.flush()

    def add_graph(self, model, input_to_model):
        self.writter.add_graph(model, input_to_model)
        self.writter.flush()

    def close(self):
        self.writter.close()


class plotwrapper(object):
    def __init__(self, plot_path):
        self.plot_path = plot_path

    