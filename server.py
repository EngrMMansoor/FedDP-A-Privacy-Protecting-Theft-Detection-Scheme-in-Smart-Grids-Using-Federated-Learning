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


fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config={"num_rounds": 2} ,
        grpc_max_message_length = 1024*1024*1024,
        strategy=strategy
    
)
