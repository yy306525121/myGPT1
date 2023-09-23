from fastapi import APIRouter

router = APIRouter()


@router.get('/')
def chat(query: str = None):
    return {'message': query}
