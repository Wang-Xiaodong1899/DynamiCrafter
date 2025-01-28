CUDA_VISIBLE_DEVICES=0 python infer_dynamic.py --val_s 0 --val_e 10 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=1 python infer_dynamic.py --val_s 10 --val_e 20 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=2 python infer_dynamic.py --val_s 20 --val_e 30 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=3 python infer_dynamic.py --val_s 30 --val_e 40 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=4 python infer_dynamic.py --val_s 40 --val_e 50 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=5 python infer_dynamic.py --val_s 50 --val_e 60 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=6 python infer_dynamic.py --val_s 60 --val_e 70 --ddim_steps 30 &
CUDA_VISIBLE_DEVICES=7 python infer_dynamic.py --val_s 70 --val_e 80 --ddim_steps 30 &
