import math
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

CLIENT_CSV_HEADER = [
    "RoundId", "ClientId", "ModelArchType", "LocalTestAccuracy", "LocalValAccuracy",
    "LocalValMicroAUC", "LocalValMacroAUC", "TestMicroAUC", "TestMacroAUC",
    "TrainingTime", "InferenceTime", "TotalDataPoints", "NewDataPoints"
]

SERVER_CSV_HEADER = [
    "RoundId", "LearningType", "TestAccuracy", "Precision", 
    "Recall", "TestMicroAUC", "TestMacroAUC", "InferenceTime"
]

def rearrange_data(dataframe, partition_list, target_column):
    """Reorder function to avoid bias in test data.

        Calling this function will ensure that at least [val + test]
        percent of the minority class samples will be present 
        in the validation and test partitions.
    """
    # count the frequency of each label
    label_counts = dataframe[target_column].value_counts()
    # create a temporary dataframe
    df_tmp = pd.DataFrame(columns=dataframe.columns)
    # calculate the number of samples per class based on the minority class
    # and the size of the validation and test partitions
    val_test_size = int(sum(partition_list[1:]) * label_counts.min())
    # for each class take val_test_size samples
    for label in dataframe[target_column].unique():
        #label_df = dataframe[dataframe[target_column] == label]
        #print(f"For label {label} I'll be taking {val_test_size} samples. Original shape {tmp_df.shape}")
        df_tmp = pd.concat([df_tmp, dataframe[dataframe[target_column] == label].iloc[:val_test_size, :]])

    # drop these samples from original dataframe
    dataframe.drop(df_tmp.index, inplace=True)
    # concatenate them again in the tail of the dataframe
    dataframe = pd.concat([dataframe, df_tmp])

    return dataframe

def ensure_representativeness(data, sample_idxs, sample_size, fill_minority=False):
    num_classes = data.unique().shape[0]
    training_data = data.loc[sample_idxs,]
    samples_per_label = sample_size // num_classes
    samples = np.array([], dtype='int64')
    for label in range(num_classes):
        label_samples = training_data[training_data == label]
        samples_per_label = min(sample_size // num_classes, label_samples.shape[0])
        if samples_per_label == 0:
            if fill_minority:
                samples_per_label = sample_size // num_classes
                samples_idxs = np.random.choice(data[data == label].index, samples_per_label, replace=True)
            else:    
                samples_idxs = np.random.choice(data[data == label].index, 1)
        else:
            if fill_minority:
                samples_per_label = sample_size // num_classes
                samples_idxs = np.random.choice(label_samples.index, samples_per_label, replace=True)
            else:
                samples_idxs = np.random.choice(label_samples.index, samples_per_label, replace=False)
        samples = np.concatenate((samples, samples_idxs), axis=0)

    np.random.shuffle(samples)
    return samples

def datapoints_loader(target, mode, exclude, batch_size):
    new_samples = np.array(list(set(target.index.tolist()) - set(exclude.tolist())))
    datapoints = ensure_representativeness(target, new_samples, sample_size=batch_size, fill_minority=mode)
    return datapoints

def count_labels_per_client(args, train_dl, logger):
    """
    Given each client and their distributions, return
    logs showing the distribution per each client for label
    """
    per_client_distribution = {}

    for i in range(args.n_clients):
        per_client_distribution[i] = dict()
        features, targets = train_dl[i]
        all_values = targets.value_counts()
        labels = targets.unique()
        logger.info(f'Class distribution on client{i}')
        for label in labels:
            per_client_distribution[i][label] = all_values.loc[label]
            if args.verbose:
                logger.info(f"Client{i} has {all_values.loc[label]} samples of '{label}' class")

    return per_client_distribution

def data_partition_split(data_path, partition_list, target_column="label"):
    """Splits data according to partition size list."""
    df = pd.read_csv(data_path)
    # df = rearrange_data(df, partition_list, target_column)

    train_df, val_df, test_df = np.split(df.sample(frac=1), 
        [int(partition_list[0]*len(df)), int(sum(partition_list[:-1])*len(df))])

    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    X_val = val_df.drop(target_column, axis=1)
    y_val = val_df[target_column]

    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    # print("Train targets:", y_train.value_counts())
    # print("Validation targets:", y_val.value_counts())
    # print("Test targets:", y_test.value_counts())

    # check if all classes are present in the partitions
    assert y_train.unique().shape[0] == y_val.unique().shape[0]
    assert y_train.unique().shape[0] == y_test.unique().shape[0]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_datapoints(data_idxs, data_distribution):
    n_labels, n_clients = data_distribution.shape
    assert len(data_idxs) == n_labels
    
    label_freqs = np.asarray([len(data_idxs[label]) for label in range(n_labels)])
    shard_sizes = np.rint(data_distribution * label_freqs.reshape(n_labels, 1)).astype("int64")

    datapoints_map = {i: np.array([], dtype='int64') for i in range(n_clients)}
    
    for label in range(n_labels):
        rand_label_id = np.random.choice(data_idxs[label], 1)
        for i in range(n_clients):
            # assert len(data_idxs[label]) > 0, f"There's no example of {label} label"
            if len(data_idxs[label]) == 0:
                idxs_shard = rand_label_id
                datapoints_map[i] = np.concatenate((datapoints_map[i], idxs_shard), axis=0)
            else:
                idxs_shard = np.random.choice(data_idxs[label], min(len(data_idxs[label]), shard_sizes[label,i]), replace=False)
                datapoints_map[i] = np.concatenate((datapoints_map[i], idxs_shard), axis=0)
                data_idxs[label] = list(set(data_idxs[label]) - set(idxs_shard))
            
    return datapoints_map


def data_partition_loader(training_data, n_clients, dirichlet_alpha=1e5):
    features, targets = training_data
    class_idxs = {}
    for i, label in enumerate(targets):
        if not label in class_idxs:
            class_idxs[label] = []
        class_idxs[label].append(features.index[i])

    params = np.ones(n_clients)*dirichlet_alpha
    dirichlet_distrib = np.random.dirichlet(params, targets.unique().shape[0])

    if dirichlet_alpha <= 0.1: # to avoid the exclusion of minority classes
        dirichlet_distrib = np.apply_along_axis(softmax, 1, dirichlet_distrib) 

    dp_map = split_datapoints(class_idxs, dirichlet_distrib)

    data_dl = {i: (features.loc[dp_map[i],], targets.loc[dp_map[i],].astype('int')) for i in range(n_clients)}   

    return data_dl

def softmax(w, T=0.15):
    S = sum([math.exp(x/T) for x in w])
    return [math.exp(x/T)/S for x in w]

def create_logdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return os.path.abspath(dir_path)

def save_reports(dict_log, filepath):
    with open(filepath, "w") as f:
        header = list(dict_log)
        f.write(",".join(header) + "\n")
        for i in range(len(dict_log[header[0]])):
            row = []
            for key in header:
                row.append(f"{dict_log[key][i]:.5f}" if isinstance(dict_log[key][i], float) else f"{dict_log[key][i]}")
            f.write(",".join(row) + "\n")
        f.close()