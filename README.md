# NeuralNetworking
Neural Network for ECG data gathering on small Arm architectures. 

This read me is meant to quickly introduce the use of this neural network code.

In the future, installing methods should be described and available here.
Also, some experience with the Eigen c++ library is necessary to follow this example.


How to use the network currently (as of March 26, 2016)

###Network constructor

Network neun(RowVectorXi sizes, int nols)  


sizes is a RowVectorXi that contains the number of nodes in each layer of the network.

nols is the size of RowVectorXi and contains the number of layers in the network.

Here is a example implementation of the arguments to pass into the network constructor.
int nols = 3;  
RowVectorXi sizes(nols);  

sizes(0) = 3;  
sizes(1) = 6;  
sizes(2) = 3;  


Create a matrix for the inputs and outputs of the network. For example, to solve the
xor problem.


MatrixXd inputs(4, 2);  
inputs << 0, 0,  
		  0, 1,  
		  1, 0,  
		  1, 1;  

MatrixXd targets(4, 1);  
targets << 0,  
<prev>	   1,  
		   1,  
		   0;  


Here, the target should be the output values the network will train on.


Create a matrix to hold the results. Note the the results is stored as a column vector.

MatrixXd results(3, 1);  


Next train the network using the train method.

MatrixXd train(MatrixXd inputs, MatrixXd targets)  

here is a example of using the train algorithm  
Note that the results of feedforward are outputted from the train algorithm.  


for(int i = 0; i < number_of_training_runs; i++)  
{  
   for(int x = 0; x < inputs.rows(); x++)  
	{  
		neun.train(inputs.row(x), targets.row(x) );  
		neun.updateWeights();  
	}
  
}


Also, the results of the neural network can be gather from feedforward()  

for(int c = 0; c < inputs.rows(); c++)  
{  
	results = neun.feedforward(inputs.row(c));  
	std::cout << "results: \n" << results << "\n";  
}  