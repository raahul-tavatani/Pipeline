import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training, file_list, ext, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=None, logger=logger)
        self.sample_file_list = file_list
        self.ext = ext

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        file_path = self.sample_file_list[index]
        if self.ext == '.bin':
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        else:
            raise NotImplementedError(f"Extension {self.ext} not supported")
        
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def save_predictions(pred_dicts, save_path):
    output = {
        "boxes": pred_dicts[0]['pred_boxes'].cpu().numpy().tolist(),
        "scores": pred_dicts[0]['pred_scores'].cpu().numpy().tolist(),
        "labels": pred_dicts[0]['pred_labels'].cpu().numpy().tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)


def process_folder(model, cfg, logger, input_folder, output_folder, ext):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = list(input_folder.glob(f'*{ext}'))
    files.sort()
    logger.info(f"Processing folder {input_folder} with {len(files)} files.")

    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        file_list=files,
        ext=ext,
        logger=logger
    )

    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f"Processing file {files[idx].name} ({idx + 1}/{len(files)})")
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            output_file = output_folder / (files[idx].stem + ".json")
            save_predictions(pred_dicts, output_file)
            logger.info(f"Saved prediction to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Batch prediction for multiple test folders')
    parser.add_argument('--cfg_file', type=str, required=True, help='Config YAML file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument(
        '--test_types',
        type=str,
        nargs='+',
        default=['Azimutal_tests', 'Radial_tests', 'Trajectory_tests'],
        help='List of test types (subfolders under saved_data). Example: Azimutal_tests Radial_tests Trajectory_tests'
    )
    args = parser.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger.info("Loaded config")

    base_path = Path(r"C:\Pipeline\saved_data")

    # Create a dummy dataset with empty file list but proper class_names for model init
    dummy_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        file_list=[],
        ext='.bin',
        logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    for test_type in args.test_types:
        input_folder = base_path / test_type / 'bin'
        ext = '.bin'
        output_folder = base_path / test_type / 'pred_json'
        process_folder(model, cfg, logger, input_folder, output_folder, ext)

    logger.info("Batch prediction completed.")


if __name__ == '__main__':
    main()
