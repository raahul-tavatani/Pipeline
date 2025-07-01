import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add root directory to PYTHONPATH so pcdet and other imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        if isinstance(root_path, (str, Path)):
            root_path = Path(root_path)
        self.root_path = root_path
        self.ext = ext

        data_file_list = list(root_path.glob(f'*{self.ext}')) if root_path.is_dir() else [root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        file_path = self.sample_file_list[index]
        if self.ext == '.bin':
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(file_path)
        else:
            raise NotImplementedError(f"Unsupported file extension: {self.ext}")

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='OpenPCDet Inference Demo')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='Specify the config for the model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='Extension of your point cloud data file (.bin or .npy)')
    parser.add_argument('--output_dir', type=str, default='saved_data',
                        help='Directory to save prediction JSONs')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def save_predictions(pred_dicts, save_path):
    output = {
        "boxes": pred_dicts[0]['pred_boxes'].cpu().numpy().tolist(),
        "scores": pred_dicts[0]['pred_scores'].cpu().numpy().tolist(),
        "labels": pred_dicts[0]['pred_labels'].cpu().numpy().tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=args.data_path, ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            input_file_path = Path(demo_dataset.sample_file_list[idx])
            stem = input_file_path.stem
            output_filename = output_dir / f"{stem}.json"

            save_predictions(pred_dicts, output_filename)
            logger.info(f"Saved prediction to {output_filename.resolve()}")

    logger.info("Demo done.")


if __name__ == '__main__':
    main()
