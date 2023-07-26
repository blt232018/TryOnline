from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from uvicorn import run


from routers.tryon_tool import api_router
from loguru import logger

app = FastAPI(title='Fastapi_tools')


async def start_api():
    logger.debug('main api start --> ')


@app.on_event('shutdown')
async def close_api():
    logger.info('main api close <--')

app.add_event_handler('startup', start_api)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router)


if __name__ == "__main__":
    run("main:app", workers=1)
