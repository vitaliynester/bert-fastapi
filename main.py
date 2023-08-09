from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

import services
from models import RequestModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id, "percent": 0})


@app.post("/")
async def get_compare(data: RequestModel):
    percent = services.compare(data.text1, data.text2)
    return {
        'percent': float(percent) * 100
    }
