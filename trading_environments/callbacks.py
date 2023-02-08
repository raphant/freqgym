from typing import Optional

from stable_baselines3.common.callbacks import EvalCallback


class EventCallback(EvalCallback):
    """
    Base class for triggering callback on event.

    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: (int)
    """

    def __init__(self, callback: Optional[EvalCallback] = None, verbose: int = 0):
        super(EventCallback, self).__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.parent = self

    ...

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback()
        return True
