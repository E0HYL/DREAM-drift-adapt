#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_dataset.py
@Time    :   2023/07/26 19:49:38
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import unittest
from dream.load_data import load_data_with_hold_out


class TestDataset(unittest.TestCase):
    def test_drebin(self):
        data_name = 'drebin'
        newfamily = 0
        num_family = 8
        analyzer = 'OpcodeSeqAnalyzer'
        behavior_label = True
        X_train, (y_train, yb_train), X_test, (y_test, yb_test) = load_data_with_hold_out(data_name=data_name, newfamily=newfamily, num_family=num_family, analyzer=analyzer, behavior_label=behavior_label)
        print(f'X_train: {X_train.shape}, (y_train, yb_train): ({y_train.shape} with max {y_test.max()}, {yb_train.shape})\nX_test: {X_test.shape}, (y_test, yb_test): ({y_test.shape} with max {y_test.max()}, {yb_test.shape})')
        return X_train, (y_train, yb_train), X_test, (y_test, yb_test)
        