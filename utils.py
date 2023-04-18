def is_num(token):
    try:
        if float(token):
            is_number = True
    except:
        is_number = False
    return is_number