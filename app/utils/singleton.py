import abc


class Singleton(abc.ABCMeta, type):
    """
    类单例模式
    """
    _instance: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance[cls]
