def check_missing_value(self, data):
    # TODO : Need Refactoring
    def timestamp(index=0):
        return data[self.key][index]

    data[self.key] = pd.to_datetime(data[self.key])
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
                data = data.append({self.key: cur_time}, ignore_index=True)
                cur_time = cur_time + TIMEGAP

    # Resorting by timestamp
    logger.info(f"Checking Timegap - ({TIMEGAP}), Filled : {filled_count}")
    data = data.set_index(self.key).sort_index().reset_index()

    return data, missings



        if data is None:
            return data

        if normalize is True:
            data = self.normalize(data)

        data = self.windowing(data)
        data = torch.from_numpy(data).float()
        return data