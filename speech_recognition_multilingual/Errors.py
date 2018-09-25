class Error(Exception):
    """Base class for other exceptions"""
    pass

class DatabaseLimitError(Error):
   """If rowstart is specified but limit is not"""
   pass

class ValidateDataRequiresTestDataError(Error):
   """If rowstart is specified but limit is not"""
   pass

class ShiftLargerThanWindowError(Error):
   """If rowstart is specified but limit is not"""
   pass

class TrainDataMustBeSetError(Error):
   """If rowstart is specified but limit is not"""
   pass

class EmptyDataSetError(Error):
   """If rowstart is specified but limit is not"""
   pass
