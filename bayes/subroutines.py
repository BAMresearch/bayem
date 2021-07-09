def len_or_one(obj):
    """Returns the length of an object or 1 if no length is defined."""
    if hasattr(obj, '__len__'):
        length = len(obj)
    else:
        length = 1
    return length
