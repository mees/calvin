class CalvinBaseModel:
    """
    Base class for all models that can be evaluated on the CALVIN challenge.
    If you want to evaluate your own model, implement the class methods.
    """

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
