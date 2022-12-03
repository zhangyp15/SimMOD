from mmcv.runner import HOOKS, Hook

import pdb

@HOOKS.register_module()
class SetEpoch(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        runner.model.module.img_roi_head.epoch = runner.epoch
        runner.model.module.img_roi_head.max_epochs = runner.max_epochs

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        runner.model.module.img_roi_head.iter = runner.iter
        runner.model.module.img_roi_head.max_iters = runner.max_iters

    def after_iter(self, runner):
        pass