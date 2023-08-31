from splitter.K_foldSplitter import KSplitter
from splitter.RandomSplitter import RSplitter

from model.CARTDecisionTree import CARTDecisionTree
from model.RandomForest import RandomForest
from model.NaiveBayesClassifier import NaiveBayesClassifier
from model.LinearRegression import LinearRegression
from model.K_means import KMeans
from model.KNN import KNearestNeighbors
from model.SVM import SVC
from model.GBDT import GBDTRegressor
from model.GBDT import GBDTClassifier
from model.logistic import LogisticRegression

from dataset.BostonDataset import BostonDataset
from dataset.BrecancerDataset import BrecancerDataset
from dataset.DiabetesDataset import DiabetesDataset
from dataset.DigitsDataset import DigitsDataset
from dataset.IrisDataset import IrisDataset
from dataset.WineDataset import WineDataset
import numpy,pandas

from performances.Accuracy import accuracy
from performances.F1_score import F1_score
from performances.Mae import mae
from performances.Mse import mse
from performances.Precision import precision
from performances.R2_score import R2_score
from performances.Recall import recall
from performances.Variance import variance

class Register:
    def __init__(self) -> None:
        self.cla_performances={
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": F1_score
            
        }
        self.reg_performances={
            "Variance":variance,
            "Mae": mae,
            "Mse": mse,
            "R2_score":R2_score
        }
        self.cla_datasets={
            'iris':IrisDataset,
            'digits':DigitsDataset,
            'breast_cancer':BrecancerDataset,
            'wine':WineDataset
        }
        self.reg_datasets={
            'diabetes':DiabetesDataset,
            'boston_housing':BostonDataset,
        }
        self.cla_model={
            'NaiveBayesClassifier':NaiveBayesClassifier,
            'CARTDecisionTree':CARTDecisionTree,
            'RandomForest':RandomForest,
            'KNearestNeighbors':KNearestNeighbors,
            'KMeans':KMeans,
            'SVM':SVC,
            'GBDT':GBDTClassifier,
            'LogisticRegression':LogisticRegression
        }
        self.reg_model={
            'LinearRegression':LinearRegression,
            'CARTDecisionTree':CARTDecisionTree,
            'RandomForest':RandomForest,
            'GBDT':GBDTRegressor, 
        }
        self.splitter={
            "Random_Splitter":RSplitter,
            "K-Fold_Splitter":KSplitter,
        }
