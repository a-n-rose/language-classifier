class Error(Exception):
    """Base class for other exceptions"""
    pass

class DatabaseLimitError(Error):
   
   pass

class ValidateDataRequiresTestDataError(Error):
   
   pass

class ShiftLargerThanWindowError(Error):
   
   pass

class TrainDataMustBeSetError(Error):
   
   pass

class EmptyDataSetError(Error):
   
   pass

class MFCCdataNotFoundError(Error):
   pass
