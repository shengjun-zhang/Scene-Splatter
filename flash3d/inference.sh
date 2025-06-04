python inference.py \ 
    hydra.run.dir='./' \ 
    hydra.job.chdir=true   \
    +experiment=layered_re10k  \
    +dataset.crop_border=true  \
    dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt 
    model.depth.version=v1 \
    ++eval.save_vis=True