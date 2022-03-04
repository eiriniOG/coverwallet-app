from src.accvalue_app import AccValueApp
from fastapi import FastAPI, Form, UploadFile

app = FastAPI()
app_cw = AccValueApp()

@app.get("/")
async def root():
    return {"Hi! This is an awesome api to make Account Value predictions"}
    
@app.post("/predict")
async def root(accountsFile: UploadFile, quotesFile: UploadFile, modelName: str = Form(...)):
    res = app_cw.run(modelName, accountsFile.file, quotesFile.file)
    return res.to_json()