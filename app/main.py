import multiprocessing

import gradio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config

from app.core.config import settings
from app.webui.gradio_ui import gradio_ui

App = FastAPI(title=settings.PROJECT_NAME, openapi_url=f'{settings.API_V1_PREFIX}/openapi.json')
App.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Server = uvicorn.Server(
    Config(App, host=settings.HOST, port=settings.PORT, reload=settings.DEV, workers=multiprocessing.cpu_count()))


# 路由
def init_routers():
    from app.api.api_v1 import api_router
    App.include_router(api_router, prefix=settings.API_V1_PREFIX)


def init_gradio():
    global App
    App = gradio.mount_gradio_app(App, gradio_ui(), path='/gradio')


def init_env():
    if not load_dotenv(dotenv_path=settings.config_path / '.env'):
        print('加载.env文件失败，1：.env文件是否存。2：.env文件是否为空')
        exit(1)


def start_server():
    # 加载环境变量
    init_env()
    init_gradio()
    # 加载路由
    init_routers()


if __name__ == '__main__':
    start_server()
    Server.run()
