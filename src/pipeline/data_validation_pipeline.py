from typing import Union
from src.exception import CustomError
import sys
import pandas as pd
class CustomData:
    def __init__(self,
                 mssubclass: Union[int, None] = None,
                 mszoning: Union[str, None] = None,
                 lotfrontage: Union[float, None] = None,
                 lotarea: Union[int, None] = None,
                 street: Union[str, None] = None,
                 alley: Union[str, None] = None,
                 lotshape: Union[str, None] = None,
                 landcontour: Union[str, None] = None,
                 utilities: Union[str, None] = None,
                 lotconfig: Union[str, None] = None,
                 landslope: Union[str, None] = None,
                 neighborhood: Union[str, None] = None,
                 condition1: Union[str, None] = None,
                 condition2: Union[str, None] = None,
                 bldgtype: Union[str, None] = None,
                 housestyle: Union[str, None] = None,
                 overallqual: Union[int, None] = None,
                 overallcond: Union[int, None] = None,
                 yearbuilt: Union[int, None] = None,
                 yearremodadd: Union[int, None] = None,
                 roofstyle: Union[str, None] = None,
                 roofmatl: Union[str, None] = None,
                 exterior1st: Union[str, None] = None,
                 exterior2nd: Union[str, None] = None,
                 masvnrtype: Union[str, None] = None,
                 masvnrarea: Union[float, None] = None,
                 exterqual: Union[str, None] = None,
                 extercond: Union[str, None] = None,
                 foundation: Union[str, None] = None,
                 bsmtqual: Union[str, None] = None,
                 bsmtcond: Union[str, None] = None,
                 bsmtexposure: Union[str, None] = None,
                 bsmtfintype1: Union[str, None] = None,
                 bsmtfinsf1: Union[float, None] = None,
                 bsmtfintype2: Union[str, None] = None,
                 bsmtfinsf2: Union[float, None] = None,
                 bsmtunfsf: Union[float, None] = None,
                 totalbsmtsf: Union[float, None] = None,
                 heating: Union[str, None] = None,
                 heatingqc: Union[str, None] = None,
                 centralair: Union[str, None] = None,
                 electrical: Union[str, None] = None,
                 firstflrsf: Union[int, None] = None,       # 1stFlrSF
                 secondflrsf: Union[int, None] = None,      # 2ndFlrSF
                 lowqualfinsf: Union[int, None] = None,
                 grlivarea: Union[int, None] = None,
                 bsmtfullbath: Union[float, None] = None,
                 bsmthalfbath: Union[float, None] = None,
                 fullbath: Union[int, None] = None,
                 halfbath: Union[int, None] = None,
                 bedroomabvgr: Union[int, None] = None,
                 kitchenabvgr: Union[int, None] = None,
                 kitchenqual: Union[str, None] = None,
                 totrmsabvgrd: Union[int, None] = None,
                 functional: Union[str, None] = None,
                 fireplaces: Union[int, None] = None,
                 fireplacequ: Union[str, None] = None,
                 garagetype: Union[str, None] = None,
                 garageyrblt: Union[float, None] = None,
                 garagefinish: Union[str, None] = None,
                 garagecars: Union[float, None] = None,
                 garagearea: Union[float, None] = None,
                 garagequal: Union[str, None] = None,
                 garagecond: Union[str, None] = None,
                 paveddrive: Union[str, None] = None,
                 wooddecksf: Union[int, None] = None,
                 openporchsf: Union[int, None] = None,
                 enclosedporch: Union[int, None] = None,
                 threessnporch: Union[int, None] = None,    # 3SsnPorch
                 screenporch: Union[int, None] = None,
                 poolarea: Union[int, None] = None,
                 poolqc: Union[str, None] = None,
                 fence: Union[str, None] = None,
                 miscfeature: Union[str, None] = None,
                 miscval: Union[int, None] = None,
                 mosold: Union[int, None] = None,
                 yrsold: Union[int, None] = None,
                 saletype: Union[str, None] = None,
                 salecondition: Union[str, None] = None):

        self.mssubclass = mssubclass
        self.mszoning = mszoning
        self.lotfrontage = lotfrontage
        self.lotarea = lotarea
        self.street = street
        self.alley = alley
        self.lotshape = lotshape
        self.landcontour = landcontour
        self.utilities = utilities
        self.lotconfig = lotconfig
        self.landslope = landslope
        self.neighborhood = neighborhood
        self.condition1 = condition1
        self.condition2 = condition2
        self.bldgtype = bldgtype
        self.housestyle = housestyle
        self.overallqual = overallqual
        self.overallcond = overallcond
        self.yearbuilt = yearbuilt
        self.yearremodadd = yearremodadd
        self.roofstyle = roofstyle
        self.roofmatl = roofmatl
        self.exterior1st = exterior1st
        self.exterior2nd = exterior2nd
        self.masvnrtype = masvnrtype
        self.masvnrarea = masvnrarea
        self.exterqual = exterqual
        self.extercond = extercond
        self.foundation = foundation
        self.bsmtqual = bsmtqual
        self.bsmtcond = bsmtcond
        self.bsmtexposure = bsmtexposure
        self.bsmtfintype1 = bsmtfintype1
        self.bsmtfinsf1 = bsmtfinsf1
        self.bsmtfintype2 = bsmtfintype2
        self.bsmtfinsf2 = bsmtfinsf2
        self.bsmtunfsf = bsmtunfsf
        self.totalbsmtsf = totalbsmtsf
        self.heating = heating
        self.heatingqc = heatingqc
        self.centralair = centralair
        self.electrical = electrical
        self.firstflrsf = firstflrsf
        self.secondflrsf = secondflrsf
        self.lowqualfinsf = lowqualfinsf
        self.grlivarea = grlivarea
        self.bsmtfullbath = bsmtfullbath
        self.bsmthalfbath = bsmthalfbath
        self.fullbath = fullbath
        self.halfbath = halfbath
        self.bedroomabvgr = bedroomabvgr
        self.kitchenabvgr = kitchenabvgr
        self.kitchenqual = kitchenqual
        self.totrmsabvgrd = totrmsabvgrd
        self.functional = functional
        self.fireplaces = fireplaces
        self.fireplacequ = fireplacequ
        self.garagetype = garagetype
        self.garageyrblt = garageyrblt
        self.garagefinish = garagefinish
        self.garagecars = garagecars
        self.garagearea = garagearea
        self.garagequal = garagequal
        self.garagecond = garagecond
        self.paveddrive = paveddrive
        self.wooddecksf = wooddecksf
        self.openporchsf = openporchsf
        self.enclosedporch = enclosedporch
        self.threessnporch = threessnporch
        self.screenporch = screenporch
        self.poolarea = poolarea
        self.poolqc = poolqc
        self.fence = fence
        self.miscfeature = miscfeature
        self.miscval = miscval
        self.mosold = mosold
        self.yrsold = yrsold
        self.saletype = saletype
        self.salecondition = salecondition

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                'mssubclass': [self.mssubclass],
                'mszoning': [self.mszoning],
                'lotfrontage': [self.lotfrontage],
                'lotarea': [self.lotarea],
                'street': [self.street],
                'alley': [self.alley],
                'lotshape': [self.lotshape],
                'landcontour': [self.landcontour],
                'utilities': [self.utilities],
                'lotconfig': [self.lotconfig],
                'landslope': [self.landslope],
                'neighborhood': [self.neighborhood],
                'condition1': [self.condition1],
                'condition2': [self.condition2],
                'bldgtype': [self.bldgtype],
                'housestyle': [self.housestyle],
                'overallqual': [self.overallqual],
                'overallcond': [self.overallcond],
                'yearbuilt': [self.yearbuilt],
                'yearremodadd': [self.yearremodadd],
                'roofstyle': [self.roofstyle],
                'roofmatl': [self.roofmatl],
                'exterior1st': [self.exterior1st],
                'exterior2nd': [self.exterior2nd],
                'masvnrtype': [self.masvnrtype],
                'masvnrarea': [self.masvnrarea],
                'exterqual': [self.exterqual],
                'extercond': [self.extercond],
                'foundation': [self.foundation],
                'bsmtqual': [self.bsmtqual],
                'bsmtcond': [self.bsmtcond],
                'bsmtexposure': [self.bsmtexposure],
                'bsmtfintype1': [self.bsmtfintype1],
                'bsmtfinsf1': [self.bsmtfinsf1],
                'bsmtfintype2': [self.bsmtfintype2],
                'bsmtfinsf2': [self.bsmtfinsf2],
                'bsmtunfsf': [self.bsmtunfsf],
                'totalbsmtsf': [self.totalbsmtsf],
                'heating': [self.heating],
                'heatingqc': [self.heatingqc],
                'centralair': [self.centralair],
                'electrical': [self.electrical],
                '1stflrsf': [self.firstflrsf],      # ✅ original column name
                '2ndflrsf': [self.secondflrsf],      # ✅ original column name
                'lowqualfinsf': [self.lowqualfinsf],
                'grlivarea': [self.grlivarea],
                'bsmtfullbath': [self.bsmtfullbath],
                'bsmthalfbath': [self.bsmthalfbath],
                'fullbath': [self.fullbath],
                'halfbath': [self.halfbath],
                'bedroomabvgr': [self.bedroomabvgr],
                'kitchenabvgr': [self.kitchenabvgr],
                'kitchenqual': [self.kitchenqual],
                'totrmsabvgrd': [self.totrmsabvgrd],
                'functional': [self.functional],
                'fireplaces': [self.fireplaces],
                'fireplacequ': [self.fireplacequ],
                'garagetype': [self.garagetype],
                'garageyrblt': [self.garageyrblt],
                'garagefinish': [self.garagefinish],
                'garagecars': [self.garagecars],
                'garagearea': [self.garagearea],
                'garagequal': [self.garagequal],
                'garagecond': [self.garagecond],
                'paveddrive': [self.paveddrive],
                'wooddecksf': [self.wooddecksf],
                'openporchsf': [self.openporchsf],
                'enclosedporch': [self.enclosedporch],
                '3ssnporch': [self.threessnporch],   # ✅ original column name
                'screenporch': [self.screenporch],
                'poolarea': [self.poolarea],
                'poolqc': [self.poolqc],
                'fence': [self.fence],
                'miscfeature': [self.miscfeature],
                'miscval': [self.miscval],
                'mosold': [self.mosold],
                'yrsold': [self.yrsold],
                'saletype': [self.saletype],
                'salecondition': [self.salecondition]
            }
            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            raise CustomError(e, sys)