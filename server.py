import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import os 

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
#import grpc 
import sys
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: VotingClassifier):
    """Return an evaluation function for server-side evaluation."""

#     # Load test data here to avoid the overhead of doing it in `evaluate` itself
#     _, (X_test, y_test) = utils.load_nsl_kdd()

#     # The `evaluate` function will be called after every round
#     def evaluate(parameters: fl.common.Weights):
#         # Update model with the latest parameters
#         utils.set_model_params(model, parameters)
#         loss = log_loss(y_test, model.predict_proba(X_test))
#         accuracy = model.score(X_test, y_test)
#         print("Eval accuracy : ", accuracy)
#         return loss, {"accuracy": accuracy}

#     return evaluate

# def get_models():
#             models = list()
#             models.append(('lr', LogisticRegression()))
#             models.append(('bayes', GaussianNB()))
#             models.append((('tr', DecisionTreeClassifier())))
#             models.append((('bg', BaggingClassifier())))
#             return models
        
# models = get_models()
# model = VotingClassifier(estimators=models, voting='soft')

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        min_fit_clients=2,
        min_eval_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )

# #grpc.insecure_channel('localhost:3926', options=(('grpc.enable_http_proxy', 0),))

fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config={"num_rounds": 2} ,
        grpc_max_message_length = 1024*1024*1024,
        strategy=strategy
    
)
