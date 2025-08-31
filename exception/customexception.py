import sys
import traceback
from typing import Optional, cast


class DocumentPortalException(Exception):
    """Base class for all exceptions in the Document Portal application."""
    def __init__(self, error_message,error_details:sys) :

        self.error_message = error_message

        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        last_tb=exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next


        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "Unknown"
        self.line_number = last_tb.tb_lineno if last_tb else -1


        if exc_type and exc_value:
            self.traceback_str= "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        
    def __str__(self):

        base= f"Error in [{self.file_name}] at line [{self.line_number}] | Message: {self.error_message}"

        if self.traceback_str:
            return f"{base} | Traceback: {self.traceback_str}"
        
        return base
    
    def __repr__(self):
        return f"DocumentPortalException(error_message={self.error_message!r}, file_name={self.file_name!r}, line_number={self.line_number!r})"
    


if __name__ == "__main__":
    
    try:
        a = 1 / 0
    except Exception as e:
        raise DocumentPortalException("Division failed", e) from e



        



        