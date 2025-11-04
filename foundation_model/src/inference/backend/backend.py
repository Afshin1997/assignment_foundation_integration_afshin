class InferenceBackend:
    """Abstract class to provide a common interface for all inference backend availables."""

    def generate_actions(self, observation):
        """
        Generates a sequence of actions.

        Args:
            observation (dict): contains the observations in either numpy or raw format (e.g. task text instruction).
        Returns:
            np.ndarray: the generated actions.
        """
        raise NotImplementedError()
