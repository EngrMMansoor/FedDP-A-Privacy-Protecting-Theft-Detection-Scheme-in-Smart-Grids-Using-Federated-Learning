import warnings
import flwr as fl
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
#import grpc 

import utils_thesis5
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error




if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = utils_thesis5.load_nsl_kdd()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils_thesis5.partition(X_train, y_train, 10)[partition_id]
    
    #Create ensemble model
    # def get_models():
    #         models = list()
    #         models.append(('lr', LogisticRegression()))
    #         models.append(('bayes', GaussianNB()))
    #         models.append((('tr', DecisionTreeClassifier())))
    #         models.append((('bg', BaggingClassifier())))
    #         return models

     # get a list of base models
    def get_models():
        models = list()
        # models.append(('lr', LogisticRegression(solver='liblinear', penalty="l1",max_iter=10,warm_start=True,)))
        models.append(('rf', rfc(n_estimators=10,verbose=0,warm_start=True)))
        models.append((('knn', KNN(n_neighbors=3,leaf_size=5))))
        models.append((('bg', BaggingClassifier(warm_start=True)))) 
        return models   
    models = get_models()
    model = VotingClassifier(estimators=models, voting='soft')


    #Create LogisticRegression Model
    # model = LogisticRegression(solver='liblinear',
    #     penalty="l1",
    #     max_iter=50,  # local epoch
    #     warm_start=True,  # prevent refreshing weights when fitting
    # )

    # Setting initial parameters, akin to model.compile for keras models
    utils_thesis5.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils_thesis5.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils_thesis5.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            #print(f"Training finished for round {config['rnd']}")
            return utils_thesis5.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils_thesis5.set_model_params(model, parameters)
            yhat2 = model.predict(X_test)

            loss = log_loss(y_test, model.predict_proba(X_test))
            # accuracy = model.score(X_test, y_test)
            #l=cross_val_score(model, X_test, y_test, cv=5, scoring='recall_macro')
            accuracy = accuracy_score(y_test, yhat2)

            p,r,f,s=precision_recall_fscore_support(y_test,yhat2,average='weighted')
            rmse=mean_squared_error(y_test, yhat2, squared=False) # RMSE
            mse= mean_squared_error(y_test, yhat2,squared=True) # MSE

            # print("Eval accuracy : ", accuracy)
            # print("Eval precision : ", p)
            # print("Eval recall : ", r)
            # print("Eval f-measure : ", f)
            print(f'acuracy ensemble {accuracy}\n',f'precision ensemble {p}\n',f'recall ensemble {r}\n',f'f-measure ensemble {f}')
            print("log loss : ", loss)
            print("RMSE : ", rmse)
            print("MSE: ", mse)

            return loss, len(X_test), {"accuracy": accuracy}
            
            
    # Start Flower client
    #grpc.insecure_channel('localhost:3926', options=(('grpc.enable_http_proxy', 0),))
    fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=MnistClient(), 
        grpc_max_message_length = 1024*1024*1024)
