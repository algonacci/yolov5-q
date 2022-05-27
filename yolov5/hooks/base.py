class BaseHook:

    trainer = None

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass
