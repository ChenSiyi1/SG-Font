
#SFSC UFSC SFUC UFUC
for option in SFUC
do
    if [ $option == 'SFSC' ];then
        rst_name = sf40sc800
        gt_path = data/imgs/Seen_TRAIN800
        output_file = score_${rst_name}.txt
    elif [ $option == 'UFSC' ]; then
        rst_name = uf40sc800
        gt_path = data/imgs/Unseen_TRAIN800
        output_file = score_${rst_name}.txt
    elif [ $option == 'SFUC' ]; then
        rst_name = sf40uc800
        gt_path = data/imgs/Seen_TEST
        output_file = score_${rst_name}.txt
    elif [ $option == 'UFUC' ]; then
        rst_name = uf40uc800
        gt_path = /data/imgs/Unseen_TEST
        output_file = score_${rst_name}.txt
    fi
    pred_path = result/models_/${rst_name}

#    CUDA_VISIBLE_DEVICES=0 python evaluate/evaluate.py \
    CUDA_VISIBLE_DEVICES=0 python evaluate/fid_score.py \
    -gt ${gt_path} \
    -pred ${pred_path} \
    -o scores/models_fid_${rst_name}.txt  #models_score_${rst_name}.txt
done