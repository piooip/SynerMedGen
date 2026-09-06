# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/mnt/vd-r5/data1/data1/Weiren/Bagel/data_bagel/brain_t12t1ce/t2i', # path of the parquet files
            'num_files': 1, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 544040, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/mnt/vd-r5/data1/data1/Weiren/Bagel/data_bagel/brain_t12t1ce/CMR_HKU',
            'num_files': 1,
            'num_total_samples': 2560000,
            "parquet_info_path": '/mnt/vd-r5/data1/data1/Weiren/Bagel/data_bagel/brain_t12t1ce/CMR_HKU/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/mnt/vd-r5/data1/data1/Weiren/data_all/VLM_all/slice',
			'jsonl_path': '/mnt/vd-r5/data1/data1/Weiren/data_all/VLM_all/code_CUHK/vlm/vlm1.jsonl',
			'num_total_samples': 2291256
		},
    },
}

