python FittingSingleImage.py --model_path "TrainedModels/model_Reso32HR.pth" \
                             --img "test_data/single_images/img_000099.png"\
                             --mask "test_data/single_images/img_000099_mask.png" \
                             --para_3dmm "test_data/single_images/img_000099_nl3dmm.pkl" \
                             --save_root "test_data/fitting_res" \
                             --target_embedding "LatentCodeSamples/*/S025_E14_I01_P02.pth"


