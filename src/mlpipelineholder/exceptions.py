class PipelineError(Exception):
    pass


class RegistrationError(PipelineError):
    pass


class ResolutionError(PipelineError):
    pass


class InspectionCopyError(ResolutionError):
    pass


class ExecutionError(PipelineError):
    pass


class PersistenceError(PipelineError):
    pass
