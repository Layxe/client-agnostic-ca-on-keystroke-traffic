import numpy as np
import pandas as pd
from typing import Dict
import os

# Variables
# ##################################################################################################

# Port of the tunnel server, this port is filtered out
# and not counted as a user connection
BORE_TUNNEL_PORT        = 7835
MAX_TIME_DIFFERENCE_FOR_LABEL_ASSIGNMENT = 60 * 10

# Functions
# ##################################################################################################

def convert_labels_csv_into_dict(csv_path: str, use_translated_port=False) -> dict:
    labels = pd.read_csv(csv_path, sep=";")
    labels_dict = {}
    port_column = "port"

    if use_translated_port:
        port_column = "translated_port"

    num_unique_ports = len(labels[port_column].unique())
    dataframe_size = len(labels)

    assert num_unique_ports == dataframe_size, f"Number of unique ports ({num_unique_ports}) does not match the dataframe size ({dataframe_size})"

    # Get unique users
    users = labels["user"].unique()

    # Assign every user a unique number
    user_to_number = {user: i for i, user in enumerate(users)}

    for index, row in labels.iterrows():
        labels_dict[row[port_column]] = (user_to_number[row["user"]], row["time"])

    return labels_dict

def get_folder_paths(folder_path: str) -> list:
    """Get all the paths of the flow files in a folder.

    Args:
        folder_path (str): Path to the folder containing the flow files.

    Returns:
        list: List of paths to the flow files.
    """
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))

    return paths

def load_paths_into_df(paths: list, append_filename_to_row=False) -> pd.DataFrame:
    """For loading data use this method!

    Args:
        paths (list): List of paths to the flow files.

    Returns:
        pd.DataFrame: Complete DataFrame containing all the flow data.
    """
    df: pd.DataFrame = pd.DataFrame()
    for path in paths:
        new_df = load_flow_as_dataframe(path)

        if append_filename_to_row:
            filename = os.path.basename(path)
            filename_without_extension = os.path.splitext(filename)[0]
            new_df["filename"] = filename_without_extension

        df = pd.concat([df, new_df], ignore_index=True)

    return df

def fancy_print_ip(ip_int: int) -> str:
    """Converts an integer representation of an IP address to a human-readable format.

    Args:
        ip_int (int): Integer representation of an IP address.

    Returns:
        str: Human-readable format of the IP address.
    """

    return f"{ip_int >> 24 & 0xFF}.{ip_int >> 16 & 0xFF}.{ip_int >> 8 & 0xFF}.{ip_int & 0xFF}"

def load_flow_as_dataframe(path: str) -> pd.DataFrame:
    """Loads a flow file as a pandas DataFrame.

    Args:
        path (str): Path to the flow file.

    Returns:
        pd.DataFrame: DataFrame containing the flow data.
    """
    data = np.load(path)
    df = pd.DataFrame(data, columns=["receive_time", "bytes", "src_ip", "dst_ip", "packets", "src_port", "dst_port"])

    # Convert all into 64-bit integers
    for col in df.columns:
        df[col] = df[col].astype(np.int64)

    # Convert IP addresses to human-readable format
    df["src_ip"] = df["src_ip"].apply(fancy_print_ip)
    df["dst_ip"] = df["dst_ip"].apply(fancy_print_ip)

    df["src_port"] = df["src_port"].astype(str)
    df["dst_port"] = df["dst_port"].astype(str)

    # Drop columns with receive_time of 0
    df = df[df["receive_time"] != 0]

    df.reset_index(drop=True, inplace=True)

    return df

def get_all_non_control_sockets_to_tunnel(
        concurrent_connections_df: pd.DataFrame,
        avg_packets_per_flow_threshold: int = 50,
        verbose = False) -> pd.DataFrame:
    """Filter out control connections by checking the average packets per flow. Since our
       application server is transmitting at least at 20 Hz, we can filter out all connections
       by setting a threshold of 50 packets per flow, which corresponds to > 10 Hz.

    Args:
        concurrent_connections_df (pd.DataFrame): Flows of connection to the application server.
        avg_packets_per_flow_threshold (int, optional): Average packets to be considered application traffic. Defaults to 50.
        verbose (bool, optional): Print out more information. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """

    # Filter out control connections by checking the average packets per flow
    avg_packets_per_flow_src = concurrent_connections_df.groupby("src_port")["packets"].mean()

    if verbose:
        print(f"Average packets per flow: {avg_packets_per_flow_src}")

    avg_packets_per_flow = avg_packets_per_flow_src[avg_packets_per_flow_src > avg_packets_per_flow_threshold]

    # Filter out rows that do not have the unique ports
    concurrent_connections_df = concurrent_connections_df[
        (concurrent_connections_df["src_port"].isin(avg_packets_per_flow.index)) &
        (concurrent_connections_df["dst_port"].isin(avg_packets_per_flow.index))]

    return concurrent_connections_df


def get_unique_concurrent_connections(df, application_server_port=25565, port_count_threshold=50, verbose=False):
    """
    Get all unique user connections to the application server. Ping or temporary control traffic are filtered out,
    by checking the amount of packets per port.

    Args:
        df (pd.Dataframe): Flow dataframe
        application_server_port (int, optional): Port, the application server is on. Defaults to 25565.
        port_count_threshold (int, optional): The minimum amount of messages for a connection to be kept. Defaults to 50.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # Filter all traffic from the application server
    from_server = df.where(df["dst_port"] == str(application_server_port)).dropna()
    to_server   = df.where(df["src_port"] == str(application_server_port)).dropna()

    tmp_df = pd.concat([from_server, to_server])

    print("Total number of flows: ", len(df))
    print("Flows to/from the application server: ", len(tmp_df))

    # Get all unique destination ports
    unique_dst_ports = tmp_df["dst_port"].unique()


    # Remove ports that have less than 50 entries
    port_counts      = tmp_df["dst_port"].value_counts()
    unique_dst_ports = unique_dst_ports[port_counts[unique_dst_ports] > port_count_threshold]

    if verbose:
        print(f"Unique destination ports: {unique_dst_ports}")

    df = df[(df["dst_port"].isin(unique_dst_ports)) | (df["src_port"].isin(unique_dst_ports))]

    return df

def get_best_matching_label(port, receive_time, port_to_label_mapping):
    """For a given flow packet, find the best matching label from the dict.

    Args:
        port (int): Port of the flow
        receive_time (int): Receive time of the label
        port_to_label_mapping (dict): Dict of all labels (Every time a user connects, the port and receive time is stored)

    Returns:
        int: Best matching label, -1 if no label is found
    """
    matching_ports = []

    for key, value in port_to_label_mapping.items():
        if str(key) == str(port):
            matching_ports.append(value)

    min_label = -1
    min_diff  = MAX_TIME_DIFFERENCE_FOR_LABEL_ASSIGNMENT

    for match in matching_ports:
        label = match[0]
        label_receive_time = match[1]

        diff = abs(receive_time - label_receive_time)

        if diff < min_diff:
            min_diff = diff
            min_label = label

    return min_label


def label_flow_dataframe(
        df: pd.DataFrame,
        port_to_label_mapping: Dict[str, int],
        application_server_port=25565,
        verbose=False) -> pd.DataFrame:

    frequent_flows_to_tunnel_server = get_unique_concurrent_connections(df, application_server_port=application_server_port, verbose=verbose)
    non_control_flows               = get_all_non_control_sockets_to_tunnel(frequent_flows_to_tunnel_server, verbose=verbose)

    # Create a copy of the dataframe
    non_control_flows = non_control_flows.copy()

    # Add client_port to each traffic
    non_control_flows["client_port"] = non_control_flows["src_port"]

    for row in non_control_flows.itertuples():
        if row.client_port == str(application_server_port):
            non_control_flows.loc[row.Index, "client_port"] = row.dst_port

    # This dict stores the last receive time of each port
    prev_receive_time = {}

    # In this dict, the current label for a given port is stored. This can change if the time difference
    # is too large between two flows
    current_mappings = {}

    for row in non_control_flows.itertuples():
        target_port = row.client_port

        # This is the first time we see this port, so we need to find the best matching label
        if target_port not in prev_receive_time:
            current_label =get_best_matching_label(target_port, row.receive_time, port_to_label_mapping)
            current_mappings[target_port] = current_label

        else:
            # Check if the time difference is too large, if so we same port might be assigned to a different label
            if row.receive_time - prev_receive_time[target_port] > MAX_TIME_DIFFERENCE_FOR_LABEL_ASSIGNMENT:
                current_label = get_best_matching_label(target_port, row.receive_time, port_to_label_mapping)
                current_mappings[target_port] = current_label

        prev_receive_time[target_port] = row.receive_time

        if target_port in current_mappings:
            non_control_flows.loc[row.Index, "label"] = current_mappings[target_port]

    # Drop all rows that do not have a label
    labeled_flows = non_control_flows.where(non_control_flows["label"] != -1).dropna()
    unlabeled_flows = non_control_flows.where(non_control_flows["label"] == -1).dropna()

    print(f"Number of labeled flows: {len(labeled_flows)}")
    print(f"Number of unlabeled flows: {len(unlabeled_flows)}")

    return labeled_flows

# Data Utility Functions
# ##################################################################################################

def get_time_skips(df):
    """Get the indices and values of time skips in the flow data.

    Args:
        df (pd.DataFrame): Dataframe of the flow data.

    Returns:
        Tuple[pd.Index, pd.Series]: Tuple of indices and values of time skips.
    """

    diffs = df["receive_time"].diff()

    above_threshold_diffs = diffs[diffs > 60 * 10]
    corresponding_indices = above_threshold_diffs.index
    values = df.loc[corresponding_indices, "receive_time"]

    return corresponding_indices, values

def calculate_recording_time(df):
    """Get the total recording time of the flow data in hours.

    Args:
        df (pd.DataFrame): Dataframe of the flow data.

    Returns:
        float: Total recording time in hours, rounded to 2 decimal places.
    """

    skip_index, skip_value = get_time_skips(df)

    if len(skip_value) == 0:
        return round((df["receive_time"].max() - df["receive_time"].min()) / 60 / 60, 2)

    recording_time = 0

    for i in range(len(skip_index)):
        if i == 0:
            skip_point_value = df["receive_time"].iloc[skip_index[i] - 1]
            minimum_value = df["receive_time"].min()
            recording_time = skip_point_value - minimum_value
        else:
            recording_time += df["receive_time"].iloc[skip_index[i] - 1] - skip_value.iloc[i - 1]

    recording_time += df["receive_time"].max() - skip_value.iloc[-1]

    return round(recording_time / 60 / 60, 2)
