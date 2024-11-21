1. Install the "sklearn" library for implementing the Machine Learning model. 

2. For federated learning, this project implements the library named "flwr==0.14.0", which is the open-source federated learning framework. https://flower.dev/

2. Create the environment for these virtual clients and server.
        This project executes 'server' and 'clients' in the cmd/terminal separately, by specifying the address of the local machine on which the connection is to be established. 
    
        example:
        python server.py 5002
        python client1.py 5002


3. All the clients are similar concerning the Machine Learning model (i.e. voting classifier) and executed in the same manner with only difference in the data distribution in respective "utils" files. 
(Note: Multiples clients can be implemented by changing the data and number of clients in 'strategy' in "server.py")

4. The dataset used in this project can be found at https://u.pcloud.link/publink/show?code=kZ9MqqVZQyUilfm4RVXWfn4o6haMm4OfLQtV

5. Publication of this project can be found at https://www.mdpi.com/1996-1073/15/17/6241


