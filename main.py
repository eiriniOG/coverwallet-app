from src.accvalue_app import AccValueApp
from fastapi import FastAPI, Form, UploadFile

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
    
@app.post("/predict")
async def root(accountsFile: UploadFile, quotesFile: UploadFile, model: str = Form(...)):
    app = AccValueApp(model, accountsFile.file, quotesFile.file)
    res = app.run()
    return res.to_json()