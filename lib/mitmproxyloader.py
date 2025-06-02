import pandas as pd

def __load_log_file_as_dataframe(file_path):
    import os

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        dataframe_dict = []

        for line in lines:
            split_line = line.split('<|>')
            body = split_line[0]
            receive_time = split_line[1]

            dataframe_dict.append({'receive_time': receive_time, 'body': body})

        dataframe = pd.DataFrame(dataframe_dict)

        dataframe['receive_time'] = dataframe['receive_time'].apply(lambda x: int(x))

        dataframe['bytes'] = dataframe['body'].apply(lambda x: len(x))

        # Drop first few packets
        dataframe = dataframe.iloc[10:]

        dataframe['receive_time_us'] = dataframe['receive_time'].apply(lambda x: int(x) / 1000)
        dataframe['receive_time_us'] = pd.to_datetime(dataframe['receive_time_us'], unit='us')


        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]

        dataframe['filename'] = file_name_no_ext

        return dataframe


def load_mitm_packets_as_dataframe(folder):
    """Call this method pointing to a folder with the .log files

    Args:
        folder (str): Path to the folder containing the .log files

    Returns:
        dataframe: Dataframe containing all the packets from the .log files
    """
    import os

    files           = os.listdir(folder)
    total_dataframe = pd.DataFrame()

    for file in files:
        if file.endswith(".log"):
            dataframe = __load_log_file_as_dataframe(os.path.join(folder, file))
            total_dataframe = pd.concat([total_dataframe, dataframe])

    total_dataframe = total_dataframe.sort_values(by=['receive_time_us'])
    total_dataframe = total_dataframe.reset_index(drop=True)

    return total_dataframe