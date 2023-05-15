def is_num(token):
    is_number = False
    try:
        if float(token):
            is_number = True
    except:
        is_number = False
    return is_number