img_size=80
chara_size=50
chara=TRAIN.txt

font_basefolder=data/fonts
out_basefolder=data/imgs
mkdir $out_folder

for font_set in Source #Seen Unseen
do
    font_folder=${font_basefolder}/Font_$font_set
    out_folder=${out_basefolder}/${font_set}_TRAIN
    mkdir $out_folder

    python font2img.py  --ttf_path $font_folder \
                        --img_size $img_size \
                        --chara_size $chara_size \
                        --chara $chara \
                        --save_path $out_folder
done