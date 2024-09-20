

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task "site" \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name trans-indexing \
        --max_length 256 \
        --train_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/train.dd.pair.fulldoc.json \
        --valid_file /data/users/tangyubao/translation/filter-msdoc/trans-partdev/dev.qd.pair.json \
        --output_dir "output/translation-indexing" \
        --learning_rate 0.0005 \
        --warmup_steps 10000 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 100000 \
        --save_strategy steps \
        --dataloader_num_workers 1 \
        --save_steps 1000 \
        --save_total_limit 10 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 300 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --trie_path /data/users/tangyubao/translation/data/trie/t5-base-100k-leadpsg.pkl \
        --path_did2queryID /data/users/tangyubao/translation/filter-msdoc/trans-alldev/did2queryID.tsv




