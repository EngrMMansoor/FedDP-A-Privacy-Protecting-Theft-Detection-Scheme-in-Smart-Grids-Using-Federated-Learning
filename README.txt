1. Install the "sklearn" library for the implementing the Machine Learning model. 

2. For federated learning, this project implements the library named "flwr", that is the open source federated learning framework. https://flower.dev/

2. Create the environment for these virtual clients and server.
        This project executes 'server' and 'clients' in the cmd/terminal seperately, by specifying the address of local machine on which the connection is to be established. 
    
        example: (without qutations)
        "python server.py 5002" 
        "python client1.py 5002"

3. All the clinets are similar with respect to the Machine Learning model (i.e. voting classifier) and executed in the same manner with only difference in the data distribution in respective "utils" files. 

4. Dataset used in this project can be found at https://u.pcloud.link/publink/show?code=kZ9MqqVZQyUilfm4RVXWfn4o6haMm4OfLQtV

5. Publication of this project can be found at https://www.mdpi.com/1996-1073/15/17/6241


