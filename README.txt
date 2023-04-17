1. create the environment for these virtual clients and server
2. install the necessary libraries. This project is executed using the library named "flwr" (flower federated learning).
3. server and clients are executed in the cmd/terminal seperately, by specifyin the address on which the connection is to be established. 
example:
us the following command: "python server.py 5002" and "python client_thesis1.py 5002".

4. all the clinets are similar with same voting classifier and executed in the same manner with only difference in the data distribution in respective "utils" file. 
