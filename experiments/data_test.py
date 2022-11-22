from core.data import DatasetConfig
from core.data import DataLoader


def test_local(debug=False):
    data_config = DatasetConfig(base_dir='E://nutrition5k_dataset',
                                image_dir='/imagery/realsense_overhead/',
                                splits_dir="/dish_ids/splits",
                                metadata_dir="/metadata")
    data_loader = DataLoader(data_config=data_config, debug=debug)


test_local(True)
