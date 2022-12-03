_base_ = [
    "./simmod_r101.py",
]

# using fcos3d pretraining
load_from = "ckpts/r101_dcn_fcos3d_pretrain_process.pth"