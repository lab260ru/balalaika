# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
from .nisqa.NISQA_model import nisqaModel
import argparse

def run_nisqa_with_config(config):
    if 'mode' not in config:
        raise ValueError('mode is required in config')
    if 'pretrained_model' not in config:
        raise ValueError('pretrained_model is required in config')

    if config['mode'] == 'predict_file':
        if 'deg' not in config or config['deg'] is None:
            raise ValueError('deg argument with path to input file needed')
    elif config['mode'] == 'predict_dir':
        if 'data_dir' not in config or config['data_dir'] is None:
            raise ValueError('data_dir argument with folder with input files needed')
    elif config['mode'] == 'predict_csv':
        if 'csv_file' not in config or config['csv_file'] is None:
            raise ValueError('csv_file argument with csv file name needed')
        if 'csv_deg' not in config or config['csv_deg'] is None:
            raise ValueError('csv_deg argument with csv column name of the filenames needed')
        if 'data_dir' not in config or config['data_dir'] is None:
            config['data_dir'] = ''
    else:
        raise NotImplementedError('mode given not available')

    config['tr_parallel'] = True
    config['tr_bs_val'] = config.get('bs', 1)
    config['tr_num_workers'] = config.get('num_workers', 0)
    
    nisqa = nisqaModel(config)
    nisqa.predict()