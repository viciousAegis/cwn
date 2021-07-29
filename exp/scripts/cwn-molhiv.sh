#!/bin/bash

python -m exp.run_mol_exp \
  --device 0 \
  --start_seed 0 \
  --stop_seed 9 \
  --exp_name cwn-molhiv \
  --dataset MOLHIV \
  --model ogb_embed_sparse_cin \
  --use_coboundaries True \
  --indrop_rate 0.0 \
  --drop_rate 0.5 \
  --drop_position lin2 \
  --nonlinearity relu \
  --readout mean \
  --final_readout sum \
  --lr 0.0001 \
  --graph_norm id \
  --lr_scheduler None \
  --num_layers 2 \
  --emb_dim 64 \
  --batch_size 128 \
  --epochs 150 \
  --num_workers 4 \
  --preproc_jobs 32 \
  --task_type bin_classification \
  --eval_metric ogbg-molhiv \
  --max_dim 2  \
  --max_ring_size 6 \
  --init_method sum \
  --train_eval_period 10 \
  --use_edge_features \
  --dump_curves
