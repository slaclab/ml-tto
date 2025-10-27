class NotReadyError(RuntimeError):
    pass


class TransmissionError(RuntimeError):
    pass


class NoBeamError(RuntimeError):
    pass


class BackgroundMismatchError(RuntimeError):
    pass
