from preprocessing.datacleaner import Cleaner
from models.trainModel import Predictor

cleanData = Cleaner()
cleanData = cleanData.dataCleaner()

predictor = Predictor(cleanData)
predictor.predict()

