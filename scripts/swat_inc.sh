python train.py \
    -project base \
    -dataset swat \
    -dataroot data/ \
    -model_dir '../TEEN/checkpoint/swat/teen/ft_cos-avg_cos-data_init-start_0/0422-16-12-10-332-Epo_100-Bs_256-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_600-NormF-T_16.00-tw_16.0-0.1-soft_proto/session0_max_acc.pth' \
    -base_mode 'ft_dot' \
    -new_mode 'ft_cos' \
    -lr_base 0.1 \
    -decay 0.0005 \
    -epochs_base 0 \
    -batch_size_base 128 \
    -test_batch_size 128 \
    -schedule Cosine \
    -gpu '1' \
    -temperature 16 \
    -fraction_to_keep 0.3 \
    -seed 3407