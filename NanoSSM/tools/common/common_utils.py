import random
import sys
import time

import numpy as np

INFO = "Info"
WARNING = "Warning"
ERROR = "Error"

def get_format_time(current_time, format="%Y-%m-%d %H:%M:%S"):
    now = int(current_time)
    time_array = time.localtime(now)
    format_time = time.strftime(format, time_array)
    return format_time

def get_format_now_time():
    now = time.time()
    return get_format_time(now)

def common_log(content, level=INFO):
    now_time = get_format_now_time()
    text = f"\n{now_time}\t[{level}]\t{content}"
    sys.stderr.write(text)
