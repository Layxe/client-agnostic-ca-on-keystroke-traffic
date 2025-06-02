# Imports
# ##################################################################################################

import pandas as pd
import multiprocessing as mp

# Database loading
# ##################################################################################################

def load_database_content(path: str, recorded_flow: pd.DataFrame) -> pd.DataFrame:
    """Load the database content from the given path and label the packets by using the recorded flow.

    Args:
        path (str): Path to the .csv file containing the database content.
        recorded_flow (pd.DataFrame): Dataframe containing the flow, recorded at the same time as
                                      the database content

    Returns:
        pd.Dataframe: DataFrame containing the database content with the packets labeled by the recorded flow.
    """

    # Read all packets from the .csv file
    raw_database_content = pd.read_csv(path)

    # ------------ Clean up data into from and to server packages ------------ #
    raw_database_content = raw_database_content.sort_values(by='receive_time_ms')
    raw_database_content = raw_database_content.reset_index(drop=True)

    # Split the data into packets from and to the multiplexer. This is needed, since
    # GoFlow also records the flow in both directions.
    to_frame = raw_database_content[raw_database_content['src_port'] == 12000]
    from_frame = raw_database_content[raw_database_content['dst_port'] == 12000]

    # Rename columns
    to_frame = to_frame.rename(columns={'dst_port': 'port'})
    from_frame = from_frame.rename(columns={'src_port': 'port'})

    to_frame['is_to'] = 1
    from_frame['is_to'] = 0

    # Combine frames
    combined_frame = pd.concat([to_frame, from_frame])

    combined_frame = combined_frame.sort_values(by='receive_time_ms')
    combined_frame = combined_frame.reset_index(drop=True)

    # Remove src_port and dst_port columns, since they are not needed anymore
    combined_frame = combined_frame.drop(columns=['src_port', 'dst_port'])

    # --------------- Label packets by using the recorded flow --------------- #
    port_to_label_mapping = {}

    first_port_packages = recorded_flow.groupby('client_port').first()

    for index, row in first_port_packages.iterrows():
        # Take the src_port, since we are looking at the recorded flow and the first package
        # is always from the client to the server. So the src_port is the wanted port.
        port = int(row['src_port'])
        port_to_label_mapping[port] = row['label']

    combined_frame['label'] = combined_frame['port'].map(port_to_label_mapping)

    final_frame = combined_frame.dropna()

    return final_frame

# Flow generation
# ##################################################################################################

def __group_subframe_into_flows(subframe, netflow_sampling_interval_s):

    total_flows = []

    current_time    = 0
    current_packets = 0
    current_size    = 0
    start_time      = subframe.iloc[0]['receive_time_ms']
    label = subframe.iloc[0]['label']
    to_server = subframe.iloc[0]['is_to']

    for index, row in subframe.iterrows():

        last_time = current_time
        current_time = row['receive_time_ms']

        if current_time - start_time >= netflow_sampling_interval_s * 1000:
            total_flows.append([label, current_packets, current_size, last_time - start_time, to_server, last_time])
            current_packets = 1
            current_size = row['packet_size']
            start_time = current_time
        else:
            current_packets += 1
            current_size += row['packet_size']
            current_time = row['receive_time_ms']

    return total_flows

def convert_database_frame_into_flows(frame, netflow_sampling_interval_s):

    labels = frame['label'].unique()

    subframes = []

    for label in labels:
        label_frame = frame[frame['label'] == label]
        to_frame = label_frame[label_frame['is_to'] == 1]
        from_frame = label_frame[label_frame['is_to'] == 0]

        subframes.append(to_frame)
        subframes.append(from_frame)

    total_flows = []

    with mp.Pool(mp.cpu_count()) as pool:
        flows = pool.starmap(__group_subframe_into_flows, [(subframe, netflow_sampling_interval_s) for subframe in subframes])

    for flow in flows:
        total_flows += flow


    flows_dataframe = pd.DataFrame(total_flows, columns=['label', 'packets', 'bytes', 'duration', 'to_server', 'receive_time_ms'])
    flows_dataframe['receive_time'] = flows_dataframe['receive_time_ms'] / 1000
    # Round the receive time to the nearest second
    flows_dataframe['receive_time'] = flows_dataframe['receive_time'].round(0)

    flows_dataframe = flows_dataframe.sort_values(by='receive_time_ms')
    flows_dataframe = flows_dataframe.reset_index(drop=True)

    return flows_dataframe