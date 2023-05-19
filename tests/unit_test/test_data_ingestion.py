from my_ml.ingest_data import fetch_housing_data, get_train_val_test_data
import os
import yaml
import logging

project_path = os.path.dirname(os.path.dirname(os.getcwd()))
log_path = os.path.join(project_path, "logs", "app.log")

logging.basicConfig(
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
config_file = os.path.join(project_path, "config", "housing.yml")

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

git_url = config["git_url"]
dataset_path = os.path.join(project_path, config["dataset_out_path"])
split_data_path = os.path.join(project_path, config["split_data_path"])

data_result = fetch_housing_data(git_url, dataset_path)
if data_result:
    logging.info("Data has been downloaded in {}".format(dataset_path))
else:
    logging.error("Data download has failed")

split_result = get_train_val_test_data(dataset_path, split_data_path)
if split_result:
    logging.info("Data has been splitted and saved in {}".format(split_data_path))
else:
    logging.error("Data splitting failed")
