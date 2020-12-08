

def print_progress_bar(it, total, prefix="", suffix="", dec=1, bar_len=50, fill="█"):
    """
    Call in a loop to create terminal progress bar
    @params:
        it      - Required  : current iteration (Int)
        total   - Required  : total iterations (Int)
        prefix  - Optional  : prefix string (Str)
        suffix  - Optional  : suffix string (Str)
        dec     - Optional  : positive number of decimals in percent complete (Int)
        bar_len - Optional  : character length of bar (Int)
        fill    - Optional  : bar fill character (Str)
    """
    percent = ("%" + str(dec + 4) + "." + str(dec) + "f") % (100.0 * (it / float(total)))
    filled_len = int(bar_len * it // total)
    bar = fill * filled_len + "-" * (bar_len - filled_len)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="")  # if it != total else "\n")
