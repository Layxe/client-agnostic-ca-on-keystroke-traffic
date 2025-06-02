import pandas as pd

def time_series_split(df, num_splits=5):
    # Create splits equally for each label
    split_data = []

    DATA_LENGTH_PER_LABEL = df.groupby("label").size().min() // num_splits

    print(f"Data length per label: {DATA_LENGTH_PER_LABEL}")

    for i in range(num_splits):
        start = i * DATA_LENGTH_PER_LABEL
        end = start + DATA_LENGTH_PER_LABEL

        total_split_data = pd.DataFrame()

        for label in df["label"].unique():
            temp_df = df[df["label"] == label].reset_index(drop=True)
            temp_df = temp_df.iloc[start:end]
            total_split_data = pd.concat([total_split_data, temp_df])

        split_data.append(total_split_data)

    cross_validation_data = []

    for i, split in enumerate(split_data):
        test_split = split
        train_split = pd.concat([split for j, split in enumerate(split_data) if j != i])
        cross_validation_data.append((train_split, test_split))

    return cross_validation_data

def add_readable_time(df):
    df["time"] = pd.to_datetime(df["receive_time"], unit="s")
    return df

def get_balanced_dataset(df):
    # Since we are working with time series data, we cannot sample randomy
    # Instead we will take the first n samples of each class
    n = df["label"].value_counts().min()

    balanced_df = pd.DataFrame()

    for label in df["label"].unique():
        first_n_samples = df[df["label"] == label].head(n)
        balanced_df = pd.concat([balanced_df, first_n_samples])

    balanced_size = len(balanced_df)
    full_size = len(df)

    print(f"Balanced dataset size: {balanced_size} ({balanced_size / full_size * 100:.2f}%)")

    return balanced_df