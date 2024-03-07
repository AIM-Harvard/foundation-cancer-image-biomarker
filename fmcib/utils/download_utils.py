import sys


# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    """
    Display a progress bar for a download.

    Args:
        current (int): The current progress value.
        total (int): The total progress value.
        width (int, optional): The width of the progress bar in characters. Defaults to 80.

    Raises:
        None

    Returns:
        None
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()
