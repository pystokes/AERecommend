#!/usr/bin/python3
# -*- coding: utf-8 -*-

from torch import optim

class Optimizers(object):

    @classmethod
    def get_optimizer(self, config, params):

        if config.type == 'sgd':
            return optim.SGD(params=params,
                             lr=config.lr,
                             momentum=config.momentum,
                             weight_decay=config.weight_decay)

        elif config.type == 'adagrad':
            return optim.Adagrad(params=params)

        else:
            """
            Default optimizer is Adam
            """
            return optim.Adam(params=params,
                              lr=config.lr,
                              betas=config.betas,
                              eps=config.eps,
                              weight_decay=config.weight_decay,
                              amsgrad=config.amsgrad)
        

if __name__ == '__main__':
    pass
