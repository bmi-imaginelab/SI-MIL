parent_dir='/path/to/test_dataset'

python extract_properties.py --data_path "$parent_dir/slides" --json_path "$parent_dir/Hovernet_output/json" --save_path "$parent_dir/cell_property" --workers 10

python deepzoom_tiler_organ.py --dataset "$parent_dir/slides" --save_path "$parent_dir/patches" --workers 10

python patch_dict_list.py --patch_path "$parent_dir/patches"

python extract_cell_statistics_features.py --data_path "$parent_dir/slides" --cell_properties_path "$parent_dir/cell_property" --list_dict_path "$parent_dir/patches" --save_path "$parent_dir/features/cell_statistics"  --workers 10

python extract_sna_features.py --data_path "$parent_dir/slides" --cell_properties_path "$parent_dir/cell_property" --list_dict_path "$parent_dir/patches" --save_path "$parent_dir/features/sna_statistics"  --workers 10

python extract_athena_spatial_features.py --data_path "$parent_dir/slides" --cell_properties_path "$parent_dir/cell_property" --list_dict_path "$parent_dir/patches" --save_path "$parent_dir/features/athena_statistics"  --workers 10

python extract_tissue_features.py --data_path "$parent_dir/slides" --hovernet_json_path "$parent_dir/Hovernet_output/json" --list_dict_path "$parent_dir/patches" --save_path "$parent_dir/features/tissue_statistics"  --workers 10 --background_threshold 220

python club_features.py --feat_path "$parent_dir/features" --column_name_path "$parent_dir" --list_dict_path "$parent_dir/patches" --remove_cell_type "none"

python data_filtering.py --feat_path "$parent_dir/features" --save_path "$parent_dir/features" --train_test_dict_path "$parent_dir/train_test_dict.json" --list_dict_path "$parent_dir/patches" --bins 10 --norm_feat "bin" --remove_noneoplastic "False"
