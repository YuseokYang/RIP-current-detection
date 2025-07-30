from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model.detector import RipCurrentDetector

app = FastAPI()
model = RipCurrentDetector("RCNN_checkpoint_epoch1110_dae1_46.pth")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = model.predict(image_bytes)

    return JSONResponse(content={"results": results})
