class StepRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, step_class):
        self._registry[step_class.__name__] = step_class

    def get(self, step_name):
        return self._registry.get(step_name)


step_registry = StepRegistry()
