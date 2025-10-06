import sys
import traceback

def get_custom_exception(error: Exception) -> str:
    """
    Returns a formatted string with error type, file name, line number, and message.
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_no = "<unknown>"
        
    error_message = f"[Error occurred in script: {file_name} at line {line_no} | Message: {str(error)}]"
    return error_message


class CustomException(Exception):
    """
    Custom Exception class that formats error messages with filename and line number.
    """
    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error_message = get_custom_exception(error)

    def __str__(self):
        return self.error_message
    


if __name__ == "__main__":
    try:
        x = 10 / 0
    except Exception as e:
        raise CustomException(e)

