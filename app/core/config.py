from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME = 'myGPT'
    # API路径
    API_V1_PREFIX = "/api/v1"
    # 监听HOST
    HOST = '0.0.0.0'
    # 端口
    PORT = 3000
    # 允许访问的域名
    ALLOWED_HOSTS: list = ['*']
    # 是否开发环境
    DEV: bool = True
    # 配置文件目录
    CONFIG_DIR: str = None

    @property
    def root_path(self):
        return Path(__file__).parents[2]

    @property
    def inner_config_path(self):
        return self.root_path / "config"

    @property
    def config_path(self):
        if self.CONFIG_DIR:
            return Path(self.CONFIG_DIR)
        return self.inner_config_path

    @property
    def persist_directory(self):
        return self.config_path / 'db'


settings = Settings()
