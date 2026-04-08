import sys
import src.logger
from src.exception import CustomError
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.data_validation_pipeline import CustomData
from pydantic import BaseModel
from typing import Union
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()
app.mount('/static', StaticFiles(directory='statics'), name='statics')


@app.get('/')
def root():
    return FileResponse('statics/home.html')


class PredictRequest(BaseModel):  #class for defining the input data for prediction
    mssubclass: Union[int, None] = None
    mszoning: Union[str, None] = None
    lotfrontage: Union[float, None] = None
    lotarea: Union[int, None] = None
    street: Union[str, None] = None
    alley: Union[str, None] = None
    lotshape: Union[str, None] = None
    landcontour: Union[str, None] = None
    utilities: Union[str, None] = None
    lotconfig: Union[str, None] = None
    landslope: Union[str, None] = None
    neighborhood: Union[str, None] = None
    condition1: Union[str, None] = None
    condition2: Union[str, None] = None
    bldgtype: Union[str, None] = None
    housestyle: Union[str, None] = None
    overallqual: Union[int, None] = None
    overallcond: Union[int, None] = None
    yearbuilt: Union[int, None] = None
    yearremodadd: Union[int, None] = None
    roofstyle: Union[str, None] = None
    roofmatl: Union[str, None] = None
    exterior1st: Union[str, None] = None
    exterior2nd: Union[str, None] = None
    masvnrtype: Union[str, None] = None
    masvnrarea: Union[float, None] = None
    exterqual: Union[str, None] = None
    extercond: Union[str, None] = None
    foundation: Union[str, None] = None
    bsmtqual: Union[str, None] = None
    bsmtcond: Union[str, None] = None
    bsmtexposure: Union[str, None] = None
    bsmtfintype1: Union[str, None] = None
    bsmtfinsf1: Union[float, None] = None
    bsmtfintype2: Union[str, None] = None
    bsmtfinsf2: Union[float, None] = None
    bsmtunfsf: Union[float, None] = None
    totalbsmtsf: Union[float, None] = None
    heating: Union[str, None] = None
    heatingqc: Union[str, None] = None
    centralair: Union[str, None] = None
    electrical: Union[str, None] = None
    firstflrsf: Union[int, None] = None        # 1stFlrSF
    secondflrsf: Union[int, None] = None       # 2ndFlrSF
    lowqualfinsf: Union[int, None] = None
    grlivarea: Union[int, None] = None
    bsmtfullbath: Union[float, None] = None
    bsmthalfbath: Union[float, None] = None
    fullbath: Union[int, None] = None
    halfbath: Union[int, None] = None
    bedroomabvgr: Union[int, None] = None
    kitchenabvgr: Union[int, None] = None
    kitchenqual: Union[str, None] = None
    totrmsabvgrd: Union[int, None] = None
    functional: Union[str, None] = None
    fireplaces: Union[int, None] = None
    fireplacequ: Union[str, None] = None
    garagetype: Union[str, None] = None
    garageyrblt: Union[float, None] = None
    garagefinish: Union[str, None] = None
    garagecars: Union[float, None] = None
    garagearea: Union[float, None] = None
    garagequal: Union[str, None] = None
    garagecond: Union[str, None] = None
    paveddrive: Union[str, None] = None
    wooddecksf: Union[int, None] = None
    openporchsf: Union[int, None] = None
    enclosedporch: Union[int, None] = None
    threessnporch: Union[int, None] = None     # 3SsnPorch
    screenporch: Union[int, None] = None
    poolarea: Union[int, None] = None
    poolqc: Union[str, None] = None
    fence: Union[str, None] = None
    miscfeature: Union[str, None] = None
    miscval: Union[int, None] = None
    mosold: Union[int, None] = None
    yrsold: Union[int, None] = None
    saletype: Union[str, None] = None
    salecondition: Union[str, None] = None


@app.get('/prediction_form')
def prediction_form():
    return FileResponse('statics/index.html')


@app.post('/predict')
def predict(data: PredictRequest):
    try:
        features = CustomData(
            mssubclass=data.mssubclass,
            mszoning=data.mszoning,
            lotfrontage=data.lotfrontage,
            lotarea=data.lotarea,
            street=data.street,
            alley=data.alley,
            lotshape=data.lotshape,
            landcontour=data.landcontour,
            utilities=data.utilities,
            lotconfig=data.lotconfig,
            landslope=data.landslope,
            neighborhood=data.neighborhood,
            condition1=data.condition1,
            condition2=data.condition2,
            bldgtype=data.bldgtype,
            housestyle=data.housestyle,
            overallqual=data.overallqual,
            overallcond=data.overallcond,
            yearbuilt=data.yearbuilt,
            yearremodadd=data.yearremodadd,
            roofstyle=data.roofstyle,
            roofmatl=data.roofmatl,
            exterior1st=data.exterior1st,
            exterior2nd=data.exterior2nd,
            masvnrtype=data.masvnrtype,
            masvnrarea=data.masvnrarea,
            exterqual=data.exterqual,
            extercond=data.extercond,
            foundation=data.foundation,
            bsmtqual=data.bsmtqual,
            bsmtcond=data.bsmtcond,
            bsmtexposure=data.bsmtexposure,
            bsmtfintype1=data.bsmtfintype1,
            bsmtfinsf1=data.bsmtfinsf1,
            bsmtfintype2=data.bsmtfintype2,
            bsmtfinsf2=data.bsmtfinsf2,
            bsmtunfsf=data.bsmtunfsf,
            totalbsmtsf=data.totalbsmtsf,
            heating=data.heating,
            heatingqc=data.heatingqc,
            centralair=data.centralair,
            electrical=data.electrical,
            firstflrsf=data.firstflrsf,
            secondflrsf=data.secondflrsf,
            lowqualfinsf=data.lowqualfinsf,
            grlivarea=data.grlivarea,
            bsmtfullbath=data.bsmtfullbath,
            bsmthalfbath=data.bsmthalfbath,
            fullbath=data.fullbath,
            halfbath=data.halfbath,
            bedroomabvgr=data.bedroomabvgr,
            kitchenabvgr=data.kitchenabvgr,
            kitchenqual=data.kitchenqual,
            totrmsabvgrd=data.totrmsabvgrd,
            functional=data.functional,
            fireplaces=data.fireplaces,
            fireplacequ=data.fireplacequ,
            garagetype=data.garagetype,
            garageyrblt=data.garageyrblt,
            garagefinish=data.garagefinish,
            garagecars=data.garagecars,
            garagearea=data.garagearea,
            garagequal=data.garagequal,
            garagecond=data.garagecond,
            paveddrive=data.paveddrive,
            wooddecksf=data.wooddecksf,
            openporchsf=data.openporchsf,
            enclosedporch=data.enclosedporch,
            threessnporch=data.threessnporch,
            screenporch=data.screenporch,
            poolarea=data.poolarea,
            poolqc=data.poolqc,
            fence=data.fence,
            miscfeature=data.miscfeature,
            miscval=data.miscval,
            mosold=data.mosold,
            yrsold=data.yrsold,
            saletype=data.saletype,
            salecondition=data.salecondition
        )

        df_features = features.get_data_as_dataframe() #converting into dataframe
        predict_pipeline = PredictPipeline()  #calling the predict pipeline class
        result = predict_pipeline.predict(df_features)
        
        return {'prediction': float(result[0])}

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Prediction error: {e}")
        return JSONResponse({"error": "Prediction failed"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)