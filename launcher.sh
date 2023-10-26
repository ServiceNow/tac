source activate /mnt/home/envs/p2

python trainval.py \
-e mnist \
--epochs 200 \
-sb ~/exp_data/tac_mnist \
