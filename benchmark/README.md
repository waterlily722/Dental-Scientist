# Benchmark Dataset Registry

This directory standardizes dataset metadata and split manifests for dental benchmarks.

## Included Datasets

- AlphaDent
- Dental_OPG_Xray_Dataset
- Dental_Radiography
- Dentistry_Computer_Vision_Dataset
- dental_caries_classificationv3

Each dataset folder contains:
- dataset_card.yaml
- splits.json
- data -> symlink to original data/<dataset>

## dataset_card.yaml Schema

The benchmark cards currently use this schema:

- name
- data_root
- split_file
- modality
- task_type
- label_level
- target_name
- class_names
- clinical_goal
- primary_metric
- secondary_metrics
- augmentation.allowed
- augmentation.disallowed
- imbalance.expected
- notes

Core registry consumption in [core/registry.py](core/registry.py):

- directly consumed:
	- name/data_root/split_file
	- modality/task_type/label_level/target_name
	- class_names/clinical_goal/primary_metric/secondary_metrics
	- augmentation.allowed/augmentation.disallowed
	- imbalance.expected
	- notes
- not consumed:
	- any extra keys outside this schema

## Splits.json Unified Schema

All benchmark split files use the same top-level structure:

- dataset_name
- data_root
- task_type
- class_names
- counts
- class_distribution
- splits

Where splits contains three keys:

- train
- val
- test

Each sample item under splits.* uses this unified structure:

- id
- img_path
- mask_path
- disease_dict
- tooth_dict
- structure_dict
- therapy_dict
- class_name
- label

## Field Notes

- id: unique sample id (kept from source annotation when available)
- img_path: dataset-relative path used by loaders
- mask_path: optional pixel-level mask path, empty when not provided by source data
- class_name and label: benchmark-level target label pair used by training/evaluation
- *_dict fields: task annotations from source data

The schema is unified, but dict contents are task-dependent:

- classification datasets: bbox/segmentation are typically empty lists
- detection/segmentation datasets: bbox/segmentation are filled in relevant dict entries
- tooth numbering datasets: tooth_dict contains main annotations; disease_dict can be empty

## Loader Recommendation

For current training code, read class_name/label as the primary supervised target.
Use disease_dict/tooth_dict/structure_dict/therapy_dict as auxiliary annotations depending on task.

For the dental scientist scaffold, prefer resolving dataset metadata through task_name and the core registry, then launch runs with:

- python experiment.py --task_name <task> --out_dir=run_i

This keeps dataset-specific paths, metrics, and augmentation constraints out of the training hyperparameters.

## Design Choice

Source-tracking helper metadata (such as source_file pointers) is intentionally not kept in benchmark splits,
to keep manifests simpler and focused on training-time consumption.
