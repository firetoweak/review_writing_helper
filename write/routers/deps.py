from fastapi import Request
from app.container import AppContext

def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx
