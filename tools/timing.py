#!/usr/bin/python

from time import time

def time_function(function, function_name_for_logging):
    """ 
    Time a function (0-argument wrapper) and return its result. 
    Example usage: time_function(lambda: some_other_function(100))
    """
    start = time()
    result = function()

    runtime_ms = int(1000 * (time() - start))
    print 'Function {} took {} ms to execute.'.format(function_name_for_logging, runtime_ms)

    return result
