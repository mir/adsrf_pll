def print_progress(max,current):
    percentile_count = round(max / 10) + 1
    if current % percentile_count == 0:
        print('{}%'.format(round(100 * current / max)))