from pathlib import Path


def get_absolute_path(path: str):
    """
    将路径转换为绝对路径
    """
    _p = Path(path)
    return path if _p.is_absolute() else str(_p.absolute())

# #注意根路径为 main 所在路径
# #相对路径使用 get_absolute_path 函数转化为绝对路径


MASK_OUTPUT_PATH = get_absolute_path("./routers/mask_tool/output")
MASK_MODEL_NAME = "Unet_2020-10-30"

TRYON_DATA_ROOT = get_absolute_path(
    './routers/HR_VITON_main/data/zalando-hd-resize')
TRYON_TOCG_CHECKPOINT = get_absolute_path(
    './routers/HR_VITON_main/eval_models/weights/v0.1/mtviton.pth')
TRYON_GEN_CHECKPOINT = get_absolute_path(
    './routers/HR_VITON_main/eval_models/weights/v0.1/gen.pth')
