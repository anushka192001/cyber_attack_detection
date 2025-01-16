# Defning a helper function that maps the current values of datetime features into a date feature
def date_cleaner(datetime_str):
    import numpy as np
    if datetime_str in [np.nan, "b", "0", "None"]:  # these are the missing values
        return np.nan

    if "T" in datetime_str:
        split_datetime = datetime_str.split("T")
    else:
        split_datetime = datetime_str.split()

    date = split_datetime[0]
    date_with_slash = date.replace("-", "/")

    if date_with_slash == "2002/03/20":  # this is the only instance that doesn't follow the convention
        date_with_slash = "20/03/2002"
    return date_with_slash

