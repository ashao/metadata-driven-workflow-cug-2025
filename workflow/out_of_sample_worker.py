import argparse
import pickle

import numpy as np
from smartredis import Client

def concatenate_datasets(datasets):
    KE_vert_sum = np.vstack([ds.KE_vert_sum for ds in datasets])
    RV_vert_avg = np.vstack([ds.RV_vert_avg for ds in datasets])
    slope_vert_avg = np.vstack([ds.slope_vert_avg for ds in datasets])
    Rd_dx_scaled = np.vstack([ds.Rd_dx_scaled for ds in datasets])
    EKE = np.hstack([ds.EKE for ds in datasets])

    return np.column_stack([KE_vert_sum, RV_vert_avg, slope_vert_avg, Rd_dx_scaled]), EKE

def filter_zero_eke(features, target):
    nonzero = np.log1p(target) > 0
    return features[nonzero,:], target[nonzero]

def filter_in_sample(train_ds, features, target):
    transformed_features = train_ds.transform(features)
    labels = train_ds.clusters.predict(transformed_features)
    new_samples = labels == train_ds.excluded_cluster
    out_of_sample_features = features[new_samples,:]
    out_of_sample_targets = target[new_samples]
    return out_of_sample_features, out_of_sample_targets

def main(args):
    with open(args.training_data, "rb") as f:
        train_ds = pickle.load(f)
    client = Client()

    num_new_samples = 0
    new_features_l = []
    new_target_l = []

    while True:
        while client.get_dataset_list_length_gte(args.mom6_dataset_name, 100, 50, 10):
            client.rename_list(args.mom6_dataset_name, "temp")
            datasets = client.get_dataset_list("temp")
            features, target = concatenate_datasets(datasets)
            features, target = filter_zero_eke(features, target)
            new_features, new_target = filter_in_sample(train_ds, target)
            if (n_new := len(new_target)) > 0:
                num_new_samples += n_new
                new_features_l.append(train_ds.transform(new_features))
                new_target_l.append(np.log1p(new_target))

        # Try one more time to get the list, if not, break the loop and complete
        if not client.get_dataset_list_length_gte(args.mom6_dataset_name, 100, 50, 10):
            break

    train_ds.features = np.vstack([train_ds.features] + new_features_l)
    train_ds.target = np.vstack([train_ds.target] + new_target_l)

    with open(args.output_path, "wb") as f:
        pickle.save(train_ds, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", help="Path to the training_data")
    parser.add_argument("output_path", help="Where to store out of sample data")
    parser.add_argument("mom6_dataset_name", help="The name of the dataset streamed from MOM6")

    args = parser.parse_args()
    main(args)