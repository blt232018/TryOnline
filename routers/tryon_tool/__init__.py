import random
from pathlib import Path

from aiofiles import open as aopen
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from routers.config import *
from routers.HR_VITON_main import Test

from .get_cloth_mask import GenImageMask

route_name = 'Try_On_API'
api_router = APIRouter(
    tags=[route_name],
    prefix=f'/{route_name}',)


templates = Jinja2Templates(directory="templates")
models_root = Path('./routers/HR_VITON_main/data/zalando-hd-resize/test/image')
models_set = []
current_model = ''
upload_ok = False
gen_ok = False


def set_model(reset: bool = False) -> str:
    global current_model, models_set
    if not models_set:
        models_set = [i.name for i in models_root.glob('*.jpg')]
        current_model = random.choice(models_set)
        logger.debug('set model -> {}', current_model)
    elif reset:
        current_model = random.choice(models_set)
        logger.debug('reset model -> {}', current_model)

    return current_model


mask_tool = GenImageMask(MASK_OUTPUT_PATH)
tryon_tool = Test(TRYON_DATA_ROOT, TRYON_TOCG_CHECKPOINT, TRYON_GEN_CHECKPOINT)


@api_router.on_event('startup')
def start_work():
    set_model()
    mask_tool.prepare(MASK_MODEL_NAME)


# async def stream_image(image_bytes: bytes, chunk_size=1024):
#     start = 0
#     s_len = len(image_bytes)
#     while start + chunk_size <= s_len:
#         yield image_bytes[start: start + chunk_size]
#         start += chunk_size
#     else:
#         yield image_bytes[start:]


# def array_to_bytes(image_array: np.ndarray, img_format: str = '.jpg'):
#     data = cv2.imencode(img_format, image_array)[1]
#     image_bytes = data.tobytes()
#     return image_bytes


@api_router.get('/view', response_class=HTMLResponse)
async def get_tryon(request: Request, model_name: str = Depends(set_model)):
    global gen_ok
    cloth_path = './static/tmp.jpg'
    img_array, img_mask_array = mask_tool.gen_mask(cloth_path)
    if img_array is not False:
        tryon_tool.prepare_test_loader(
            model_name, img_array, img_mask_array)
        tryon_bytes = tryon_tool.test()
        if isinstance(tryon_bytes, bytes):
            async with aopen('./static/finalimg.png', 'wb') as f:
                await f.write(tryon_bytes)
                gen_ok = True
            return templates.TemplateResponse(
                'view.html', {
                    "request": request,
                }
            )
        gen_ok = False
        img_mask_array = tryon_bytes
    raise HTTPException(400, detail=img_mask_array)


@api_router.get('/', response_class=HTMLResponse)
async def get_index(request: Request, model_name: str = Depends(set_model)):
    model_path = Path('./static/current_model.jpg')
    if not model_path.exists():  # type: ignore
        model_path.hardlink_to(models_root / model_name)

    return templates.TemplateResponse(
        'main.html', {
            "request": request,
            "model_name": model_name,
            "upload_ok": upload_ok,
            "gen_ok": gen_ok
        }
    )


@api_router.get('/reset_model', response_class=HTMLResponse)
async def reset_model(request: Request):
    global gen_ok
    model_name = set_model(True)
    model_path = Path('./static') / 'current_model.jpg'
    if model_path.exists():
        model_path.unlink(missing_ok=True)
    model_path.hardlink_to(models_root / model_name)
    gen_ok = False
    return templates.TemplateResponse(
        'main.html', {
            "request": request,
            "model_name": model_name,
            "upload_ok": upload_ok,
            "gen_ok": gen_ok
        }
    )


@api_router.post('/upload', response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile):
    global upload_ok, gen_ok
    pic_name = "tmp.jpg"
    async with aopen(f'./static/{pic_name}', 'wb') as f:
        await f.write(await file.read())
        upload_ok = True
        gen_ok = False

    return templates.TemplateResponse(
        'fileUpload_cloth.html', {
            "request": request,
            "pic_name": pic_name,
            "route": f'/{route_name}'
        }
    )
