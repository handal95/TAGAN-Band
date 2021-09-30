import pandas as pd
from utils.logger import Logger

logger = Logger(__file__)


def check_missing_value(data):
    # TODO : Need Refactoring

    key = "date"

    def timestamp(index=0):
        return data[key][index]

    data[key] = pd.to_datetime(data[key])
    TIMEGAP = timestamp(1) - timestamp(0)

    missings = list()
    filled_count = 0
    for i in range(1, len(data)):
        if timestamp(i) - timestamp(i - 1) != TIMEGAP:
            start_time = timestamp(i - 1) + TIMEGAP
            end_time = timestamp(i) - TIMEGAP

            missings.append([str(start_time), str(end_time)])

            # Fill time gap
            cur_time = start_time
            while cur_time <= end_time:
                filled_count += 1
                data = data.append({key: cur_time}, ignore_index=True)
                cur_time = cur_time + TIMEGAP

    # Resorting by timestamp
    logger.info(f"Checking Timegap - ({TIMEGAP}), Filled : {filled_count}")
    data = data.set_index(key).sort_index().reset_index()

    return data, missings


if __name__ == "__main__":
    data = pd.read_csv("data/agricultural_product_full.csv")

    data, missings = check_missing_value(data)
