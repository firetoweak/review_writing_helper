from fastapi import FastAPI

from routers import help as help_router
from routers import heuristic as heuristic_router
from routers import kb as kb_router
from routers import merge as merge_router
from routers import outline as outline_router
from routers import polish as polish_router
from routers import review as review_router
from routers import writing as writing_router


app = FastAPI(title="AI Writing Review Helper")

app.include_router(outline_router.router)
app.include_router(heuristic_router.router)
app.include_router(review_router.router)
app.include_router(help_router.router)
app.include_router(merge_router.router)
app.include_router(polish_router.router)
app.include_router(writing_router.router)
app.include_router(kb_router.router)
