import importlib

get_dataset = lambda name : importlib.import_module("src.dataset." + name)
