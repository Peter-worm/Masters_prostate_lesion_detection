def find_periods(arr):
    if len(arr) == 0:
        return []

    periods = []
    start = arr[0]
    prev = arr[0]

    for num in arr[1:]:
        if num == prev + 1:
            prev = num
        else:
            periods.append((start, prev))
            start = num
            prev = num

    periods.append((start, prev))
    return periods