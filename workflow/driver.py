import glob
import shutil
from pathlib import Path

from cmflib import cmf, cmfquery
import numpy as np
from smartsim import Experiment

DAYMAX = 2200
DAYMAX_LOW = 1096
MOM6_NPROCS = 24
DATAPATH = Path("/lustre/data/shao/cug_2025/")
RESTART_PATH = DATAPATH / "restart"
BASE_TRAINING_DATA = DATAPATH / "truncated_data.pkl"
TEST_DATA = DATAPATH / "excluded_data.pkl"
MLMD_FILE = str(Path("./MDWC2025_mlmd").resolve())
ARCHITECTURES = ["EKEResNet", "EKEBottleneckResNet"]


def clean_cmf():
    dirs = (".dvc", "cmf_artifacts", "files", "CMF-SmartSim-CUG2025")
    files = (".cmfconfig", ".dvcignore", ".gitignore", MLMD_FILE)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    for f in files:
        Path(f).unlink(missing_ok=True)
    for f in Path("./").glob("*.dvc"):
        f.unlink()


def log_mom6_inputs(run_path, metawriter):
    metawriter.log_dataset(f"{run_path}/MOM_override", "input")
    metawriter.log_dataset(f"{run_path}/INPUT/MOM.res.nc", "input")
    metawriter.log_dataset(f"{run_path}/INPUT/ocean_solo.res", "input")


def create_sampler(exp):
    # Configure the data sampler
    rs_sampler = exp.create_run_settings(
        "python",
        exe_args=[
            "out_of_sample_worker.py",
            str(BASE_TRAINING_DATA),
            f"{exp.exp_path}/archive/data_sampler",
            "MEKE_training_data",
            "/lustre/data/shao/cug_2024/model_data/MKE.nc",
            "/lustre/data/shao/cug_2024/model_data/ocean_geometry.nc",
            "80",
            f"{MOM6_NPROCS}",
            "--mlmd_path",
            MLMD_FILE,
        ],
    )
    sampler = exp.create_model("data_sampler", rs_sampler)
    sampler.attach_generator_files(to_copy="./out_of_sample_worker.py")
    return sampler


def create_retrainer(exp, architecture, device):
    sampler_path = f"{exp.exp_path}/archive/data_sampler"
    extra_files = [str(path.resolve()) for path in Path(sampler_path).glob("*.pkl")]
    rs_retrainer = exp.create_run_settings(
        "python",
        exe_args=[
            "retrainer.py",
            MLMD_FILE,
            architecture,
            str(BASE_TRAINING_DATA),
            str(TEST_DATA),
            *extra_files,
            "--device",
            device,
        ],
    )
    retrainer = exp.create_model(f"{architecture}_retrainer", rs_retrainer)
    retrainer.attach_generator_files(to_copy="./retrainer.py")
    return retrainer


def create_mom6_high_res(exp, restart_path, daymax):
    rs_mom6_high_res = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6_high_res.set_tasks(MOM6_NPROCS)
    mom6_high_res = exp.create_model(
        "MOM6-high-res",
        rs_mom6_high_res,
        params={
            "DAYMAX": daymax,
            "NIGLOBAL": 240,
            "NJGLOBAL": 320,
            "DT": 300.0,
            "DT_THERM": 900.0,
            "MEKE_TRAINING": ".true.",
            "KHTH": 0.0,
            "MEKE_KHCOEFF": 0.0,
        },
    )
    mom6_high_res.attach_generator_files(
        to_copy="MOM6_inputs/Phillips_2layer",
        to_configure=["MOM6_inputs/MOM_override"],
        to_symlink=str(restart_path),
    )
    return mom6_high_res


def create_mom6_low_res(exp, model_path):
    rs_mom6_low_res = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6_low_res.set_tasks(MOM6_NPROCS)
    mom6_low_res = exp.create_model(
        "MOM6-low-res",
        rs_mom6_low_res,
        params={
            "DAYMAX": DAYMAX_LOW,
            "NIGLOBAL": 60,
            "NJGLOBAL": 80,
            "DT": "1800.0",
            "DT_THERM": "3600.0",
            "MEKE_TRAINING": ".false.",
            "KHTH": 1000.0,
            "MEKE_KHCOEFF": 1.0,
        },
    )
    mom6_low_res.attach_generator_files(
        to_copy="MOM6_inputs/Phillips_2layer",
        to_configure=["MOM6_inputs/MOM_override"],
        to_symlink=str(RESTART_PATH / "low_res" / "INPUT"),
    )
    mom6_low_res.add_ml_model(
        "mleke", backend="TORCH", model_path=model_path, device="GPU"
    )
    return mom6_low_res

def archive_retrainer(exp, retrainer):
    archive_path = Path(exp.exp_path) / "archive" / retrainer.name
    archive_path.mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{retrainer.path}/*.pth")
    for f in files:
        shutil.copyfile(f, archive_path)

def archive_mom6(exp, mom6):
    archive_path = Path(exp.exp_path) / "archive" / "high_res" / "INPUT"
    archive_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{mom6.path}/RESTART/MOM.res.nc", archive_path)
    shutil.copy(f"{mom6.path}/RESTART/ocean_solo.res", archive_path)

def retrieve_best_model(query):
    models = query.store.get_artifacts_by_type("Model")
    test_loss = []
    for model in models:
        loss_value = (
            model.custom_properties["test_loss"].double_value
            if "test_loss" in model.custom_properties
            else np.inf
        )
        test_loss.append(loss_value)

    minidx = np.argmin(test_loss)
    return models[minidx].name.split(":")[0], test_loss[minidx]


def main():
    # Initialize cmf to track workflow artifacts and metrics
    # TODO: Create a configuration file for global variables like the pipeline name,
    # path to the CMF file, etc.
    # TODO: Enable neo4j to ensure consistency between workflow artifacts as inputs/outputs
    # TODO: Reuse exection in the C-interface
    cmf.cmf_init(
        type="local",
        path="./",
        git_remote_url="git@github.com:user/repo.git",
    )
    metawriter = cmf.Cmf(
        filepath=MLMD_FILE, pipeline_name="CMF-SmartSim-2025", graph=False
    )

    exp = Experiment("CMF-SmartSim-CUG2025", launcher="local")

    # TODO: Add custom_properties to each context/execution to define
    # configuration parameters
    metawriter.create_context(pipeline_stage="Data-generation")
    metawriter.create_execution("Simulation")

    # Create and configure the database
    db = exp.create_database(interface="lo")

    # Generate new training data
    min_train_loss = np.inf
    high_res_restart = RESTART_PATH / "high_res" / "INPUT"
    daymax = DAYMAX
    try:
        exp.start(db)

        while min_train_loss > 1.:
            # Create the sampler
            sampler = create_sampler(exp)

            mom6_high_res = create_mom6_high_res(exp, high_res_restart, daymax)
            daymax += 5

            # Generate the run directory for the high res model
            exp.generate(mom6_high_res, sampler, overwrite=True)
            log_mom6_inputs(mom6_high_res.path, metawriter)

            # while min_train_loss > 1.:
            exp.start(mom6_high_res, sampler, block=True)
            archive_mom6(exp, mom6_high_res)
            high_res_restart = Path(exp.exp_path) / "archive" / "high_res" / "INPUT"
            retrainers = [
                create_retrainer(exp, arch, f"cuda:{i}")
                for i, arch in enumerate(ARCHITECTURES)
            ]
            exp.generate(*retrainers, overwrite=True)
            exp.start(*retrainers, block=True)

            # Find the best model based on the final validation metrics
            query = cmfquery.CmfQuery(MLMD_FILE)
            model_path, min_train_loss = retrieve_best_model(query)

        mom6_low_res = create_mom6_low_res(exp, model_path)
        exp.generate(mom6_low_res, overwrite=True)
        exp.start(mom6_low_res)
    finally:
        exp.stop(db)


if __name__ == "__main__":
    clean_cmf()
    main()
