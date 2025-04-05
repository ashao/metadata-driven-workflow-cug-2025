import argparse
import pickle
from cmflib import cmf

from datetime import datetime
from pathlib import Path

import gcm_filters
import numpy as np
import xarray as xr
from smartredis import Client

INDEX_LIST_NAME = "indices"
TEST = False

# Reconstruct global array from each rank's dataset

# Convert to an xarray dataset to apply filter

# Calculate EKE

str_to_dt = lambda ds: datetime.strptime(ds.get_name().split("_")[-1], "%Y.%m%d%H%M%S")


def retrieve_datasets(client, dataset_list_name, nprocs):
    """Retrieve all available datasets and sort by timestamp
    :param client: An initialized client
    :param dataset_list_name: The name of the dataset list to process
    :return: Dictionary whose keys are the timestamp and values are
                       a list of sub-datasets from each rank
    """
    if TEST:
        datasets = client.get_datasets_from_list(dataset_list_name)
    else:
        client.rename_list(dataset_list_name, "temp")
        datasets = client.get_datasets_from_list("temp")

    print(datasets[0].get_name())
    # Get all unique timestamps
    timestamps = list(set(str_to_dt(ds) for ds in datasets))

    # Sort datasets by time stamp
    ds_by_time = {timestamp: [] for timestamp in timestamps}
    for ds in datasets:
        timestamp = str_to_dt(ds)
        ds_by_time[timestamp].append(ds)
    # Only retain timestamps that have data from every rank
    for k, v in ds_by_time.items():
        print(f"{k}:{len(v)}")
    full_ds = {k: v for k, v in ds_by_time.items() if len(v) == nprocs}
    print(f"Number of full datasets: {len(full_ds)}")
    return full_ds


def calculate_indices(ds):
    """Derive the global and local indices associated with a dataset

    :param ds: Dataset posted by the MOM6 simulation
    :return: global and local indices
    """
    iscg = ds.get_meta_scalars("idg_offset") + ds.get_meta_scalars("isc") - 1
    iecg = ds.get_meta_scalars("idg_offset") + ds.get_meta_scalars("iec")
    isc = ds.get_meta_scalars("isc") - 1
    iec = ds.get_meta_scalars("iec")

    jscg = ds.get_meta_scalars("jdg_offset") + ds.get_meta_scalars("jsc") - 1
    jecg = ds.get_meta_scalars("jdg_offset") + ds.get_meta_scalars("jec")
    jsc = ds.get_meta_scalars("jsc") - 1
    jec = ds.get_meta_scalars("jec")

    return iscg[0], iecg[0], jscg[0], jecg[0], isc[0], iec[0], jsc[0], jec[0]


def reconstruct_global(ds_by_time, field_name, ni, nj):
    """Reconstruct the global array from datasets of each rank's subdomain

    :param ds_by_time: Dictionary whose keys are the timestamp and values are
                       a list of sub-datasets from each rank
    :param field_name: The field to reconstruct
    :param ni: The size of the global array in the i-direction
    :param nj: The size of the global array in the j-direction
    :return: Globally reconstructed array
    """
    ntime = len(ds_by_time)
    global_field = np.zeros((ntime, ni, nj))

    for tidx, datasets_time in enumerate(ds_by_time.values()):
        for ds in datasets_time:
            iscg, iecg, jscg, jecg, isc, iec, jsc, jec = calculate_indices(ds)
            sub_array = ds.get_tensor(field_name)
            global_field[tidx, iscg:iecg, jscg:jecg] = sub_array[isc:iec, jsc:jec]
    return global_field


def calculate_features(ds, grid_ds, mke_ds, filter_scale):
    # Create a filter to mimic the variables from the coarse resolution

    train_ds = xr.Dataset(coords=ds.coords)
    grid_length = float(grid_ds.dxT.min())
    filter = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=grid_length,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    # Calculate EKE
    train_ds["EKE"] = ds["KE_vert_sum"] - mke_ds["KE_vert_sum"]
    train_ds["Rd_dx_scaled"] = ds["Rd_dx_scaled"] * grid_length / filter_scale

    # Coarse-grain all the other variables
    for var in ["KE_vert_sum", "RV_vert_avg", "slope_vert_avg"]:
        train_ds[var] = filter.apply(ds[var], dims=["yh", "xh"])

    return train_ds


def main(args):

    if args.mlmd_path:
        metawriter = cmf.Cmf(
            args.mlmd_path, pipeline_name="CMF-SmartSim-2025", graph=False
        )
        metawriter.create_context(pipeline_stage="Data-generation")

    filter_scale = int(args.filter_scale) * 1.0e3  # Convert to meters
    # Load all data needed for this worker
    with open(args.training_data, "rb") as f:
        training_data = pickle.load(f)
    mke_ds = xr.open_dataset(args.mke_dataset)
    grid_ds = xr.open_dataset(args.grid_dataset)

    # Initialize the SmartRedis client
    if TEST:
        client = Client(False, "127.0.0.1:6379")
    else:
        client = Client()

    # Wait until all the processors have posted their index metadata
    if client.poll_list_length(INDEX_LIST_NAME, int(args.nprocs), 50, 25):
        index_datasets = client.get_datasets_from_list(INDEX_LIST_NAME)
        print(f"Index datasets retrieved")
    else:
        raise Exception("Index datasets not available")
    # Set the size of the template
    ni = index_datasets[0].get_meta_scalars("ieg")[0]
    nj = index_datasets[0].get_meta_scalars("jeg")[0]

    iteration = 0

    while True:

        found = False
        while client.poll_list_length_gte(args.mom6_dataset_name, int(args.nprocs), 50, 200):
            print(f"Rank datasets retrieved")
            iteration += 1

            ds_by_time = retrieve_datasets(
                client, args.mom6_dataset_name, int(args.nprocs)
            )

            # Create a new xarray dataset from streamed information
            coords = mke_ds.coords
            coords["Time"] = sorted(list(ds_by_time.keys()))
            coords["Time"] = range(len(ds_by_time))
            new_ds = xr.Dataset(coords=coords)
            for var in ["slope_vert_avg", "KE_vert_sum", "Rd_dx_scaled", "RV_vert_avg"]:
                new_ds[var] = (
                    ("Time", "xh", "yh"),
                    reconstruct_global(ds_by_time, var, ni, nj),
                )
            train_ds = calculate_features(new_ds, grid_ds, mke_ds, filter_scale)
            train_ds["EKE"] = train_ds["EKE"].where(train_ds["EKE"]>0., 0.)
            train_ds = train_ds.isel(yh=slice(20,-20))
            training_data.reset_from_dataset(train_ds)
            clusters = training_data.clusters.predict(training_data.features)
            retain = clusters == training_data.excluded_cluster
            training_data.features = training_data.features[retain,:]
            training_data.target = training_data.target[retain]
            train_ds.to_netcdf(f"training_data_{iteration:03d}.nc")

            timestamp = str(sorted(list(ds_by_time))[0]).replace(" ", "_")
            out_path = Path(args.output_path)
            out_path.mkdir(parents=True, exist_ok=True)
            outfile = Path(args.output_path) / f"training_data_{timestamp}.pkl"
            print(f"New samples: {len(training_data)}")
            with open(outfile, "wb") as f:
                pickle.dump(training_data, f)

            if args.mlmd_path:
                metawriter.create_execution("Sampler")
                metawriter.log_metric(
                    "sampler_metrics", {"n_new_points": str(len(training_data))}
                )
                metawriter.log_dataset(str(Path(outfile).resolve()), "input")
            # TODO: Log this artifact with cmf OR do this as log_dataslice?
            found = True

        # Try one more time to get the list, if not, break the loop and complete
        if (not client.poll_list_length_gte(args.mom6_dataset_name, 100, 10, 10)) and (not found):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", help="Path to the training data")
    parser.add_argument("output_path", help="Where to store the sample data")
    parser.add_argument(
        "mom6_dataset_name", help="The name of the dataset streamed from MOM6"
    )
    parser.add_argument(
        "mke_dataset", help="Dataset containing the background kinetic energy"
    )
    parser.add_argument("grid_dataset", help="Dataset containing grid metrics")
    parser.add_argument("filter_scale", help="The scale to filter the features to (km)")
    parser.add_argument("nprocs", help="Number of processors used in the simulation")
    parser.add_argument(
        "--mlmd_path", help="The path to the CMF database file", required=False
    )

    args = parser.parse_args()
    print(args)
    main(args)
