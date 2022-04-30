python DataProcess/Gen_HeadMask.py --gpu_id 0 --img_dir "test_data/single_images"
python DataProcess/Gen_Landmark.py --img_dir "test_data/single_images"
python Fitting3DMM/FittingNL3DMM.py --img_size 512 \
                                    --intermediate_size 256  \
                                    --batch_size 9 \
                                    --img_dir "test_data/single_images"