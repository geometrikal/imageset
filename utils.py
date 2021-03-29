def function_wrapper(fn, num_inputs, args):
    if num_inputs == 1:
        def wrapper(x):
            return fn(x, *args)
        return wrapper
    elif num_inputs == 2:
        def wrapper(x, y):
            return fn(x, y, *args)
        return wrapper
    elif num_inputs == 3:
        def wrapper(x, y, z):
            return fn(x, y, z, *args)
        return wrapper