import os
import sys

def error_message(message,error_detail:sys):
    _,_,exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    line_number = exec_tb.tb_lineno
    error_message = f'{message} error occured in {file_name} at line {line_number}'
    return error_message



class CustomError(Exception):
    def __init__(self,message,error_level:sys):
        self.message = error_message(message,error_level)
    def __str__(self):
        return self.message

