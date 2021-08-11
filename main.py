from preprocessing.datacleaner import Cleaner
from preprocessing.trainModel import Predictor
import pandas as pd

cleanData = Cleaner()
cleanData = cleanData.dataCleaner()

predictor = Predictor(cleanData)
predictor.predict()

