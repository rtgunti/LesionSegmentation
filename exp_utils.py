# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:09:43 2020

@author: rtgun
"""
import importlib
def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module