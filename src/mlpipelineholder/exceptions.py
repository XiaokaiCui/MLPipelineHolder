class PipelineError(Exception):
    pass


class RegistrationError(PipelineError):
    pass


class ResolutionError(PipelineError):
    pass


class InspectionCopyError(ResolutionError):
    pass


class InspectionMemoryError(ResolutionError):
    """Protected inspection was rejected by the memory preflight or failed while copying."""

    def __init__(
        self,
        message: str,
        *,
        node_name: str | None = None,
        function_name: str | None = None,
        inputs: str | None = None,
        reason: str | None = None,
        copy_started: bool = False,
    ) -> None:
        super().__init__(message)
        self.node_name = node_name
        self.function_name = function_name
        self.inputs = inputs
        self.reason = reason
        self.copy_started = copy_started


class ExecutionError(PipelineError):
    pass


class PersistenceError(PipelineError):
    pass
