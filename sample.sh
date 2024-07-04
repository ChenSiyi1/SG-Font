NUM_GPUS=1

for id in {0..1};
do
  TEST_SFUC="--sty_img_path data/imgs/Seen_TRAIN800/id_${id} --img_save_path ./result/models/sf40uc800/id_${id} --gen_txt_file ./TEST.txt --num_samples 10"
  TEST_UFUC="--sty_img_path data/imgs/Unseen_TRAIN800/id_${id} --img_save_path ./result/models/uf40uc800/id_${id} --gen_txt_file ./TEST.txt --num_samples 10"

  CUDA_VISIBLE_DEVICES=0 python ./sample.py $TEST_SFUC

done

 
