#define FIRST_LAYER 1
#define MID_LAYER 2
#define LAST_LAYER 3
#include <iostream>
#include <Eigen/Dense>
#define DEBUG true

// Eigen is the name of a free c++ linear algebra library available online. 
// For use in a small microprocessor, especially one with little memory, I would not recommend Eigen.
// However, it does a good job with optimizations. For using a neural network on a piece of movable hardware, 
// I think I need to rewrite only the linear algebra commands needed for the neural network, and make
// optimizations for ARM archeticture by looking at Eigen library code, or some other equivalent.

using namespace Eigen;

// The layer class controls the nodes for the neural network.
class Layer;

// Sortof a Eigen coding convention. redefine a matrix of Layer objects with only one row
// and a varying size column and RowVectorXL. 
//
typedef Matrix<Layer, 1, Dynamic> RowVectorXL;


// Just used for debugging.
void pause()
{
	std::cout << "Press enter to continue ...";
	std::cin.get();
}

// This is the activation function for the neural network.
//
// Note that in the future, this should have alternative activation functions. (softmax??, cross entropy??)
double sigm(double x)
{
	return 1 / (1 + exp(-1*x));
}

// Definition for the derivative of the sigmoid activation function.
// The sigmoid function is a unique function in that it's derivative is equal 
// to itself multiplied by (1-itself)
//
double sigm_derivative(double x)
{
	return sigm(x) * (1 - sigm(x));
}


// The Layer class for the layers of the network
class Layer
{
	// Currently, everything is public with no get or edit functions. However, this could be changed
	// to create more modular code.
	public:
		// each layer in the network has a flag to tell if it is the 
		// first layer, middle layer, or last layer
		// Optimizations and different code should be run for different layers.
	int flags;
	MatrixXd weight_matrix; // matrix of weights for a single layer. 
				// The column length should be this layer's number of nodes
				// and the row length should be the previous layer's number of nodes
	int number_of_nodes;
				// The number of nodes in this layer.
	VectorXd bias_vector;
				// The vector holding the bias for the nodes in this layer.
	VectorXd activation_vector;
				// Vector that holds the output of this layer. holds the activation's
				// output from the activation function (currently sigmoid)
	VectorXd weighted_inputs;
				// The values that are passed into the activation functions
				// for this layer. Useful to keep a hold of these for the backpropagation 
				// algorithm.
	Layer* next_layer;
				// pointer to next layer. Should be null, if this layer is the last.
	Layer* prev_layer;
				// pointer to previous layer. should be null if this is the first layer.
	VectorXd delta;
				// delta is equivalent to the derivative of the cost function (error function)
				// with respect to the bias's in this layer.
				//
				// for the first layer it is 
				// (targets-inputs) * derivative of sigmoid(weighted_inputs for last layer).
	MatrixXd dC_dw;
				// The derivative of the cost function with respect to the weights in this layer.
				//

	// Empty Layer construction.
	// When the network is made, the layers aren't initialized to anything.
	// Then, the network runs Layer_init instead of the normal constructors.
	Layer()
	{

	}


	// Layer construction that just wraps the Layer_init function.
	Layer(int _number_of_nodes,
		       	int prev_layer_number_of_nodes,
		       	Layer* _prev_layer,
		       	Layer* _next_layer)
	{
		// call layer initializing function
		Layer_init(_number_of_nodes, 
				prev_layer_number_of_nodes, 
				_prev_layer, 
				_next_layer);
	}



	// Layer initilization.
	void 
	Layer_init(int _number_of_nodes, 
			int prev_layer_number_of_nodes, 
			Layer* _prev_layer, 
			Layer* _next_layer)
	{
		// See if this is the first layer.
		if(_prev_layer == NULL)
		{
		
			flags = FIRST_LAYER;
		} else if(_next_layer == NULL)	// See if this is the last layer.
		{

			flags = LAST_LAYER;
		} else {

			flags = MID_LAYER; 
		}
		

		number_of_nodes = _number_of_nodes;
		
		if(flags != FIRST_LAYER)
		{
			weight_matrix = MatrixXd::Random(prev_layer_number_of_nodes, number_of_nodes);
			// optionally, the first layer can contain no bias's and be nothing but a placeholder.
			// depending on the problem the neural network is trying to solve, this could be detrimmental to the solution.
			// bias_vector = MatrixXd::Random(number_of_nodes, 1);
		}

		// delta for a layer is the derivative of the cost with respect to each bias.
		// should be the same size as bias_vector.
		delta.resize(number_of_nodes, 1);

		// bias_vector for all layers. 
		// Note that if you do not want bias's for the first layer, this must be uncommented.
		bias_vector = MatrixXd::Random(number_of_nodes, 1);

		// Holds the activations output from the sigmoid for this layer. 
		activation_vector.resize(number_of_nodes, 1);

		// holds the values entered into the activation function for this layer. 
		weighted_inputs.resize(number_of_nodes, 1);

		// Set up pointers to and from the last layer appropriately
		if(flags != FIRST_LAYER)	
			prev_layer = _prev_layer;
		if(flags != LAST_LAYER)
			next_layer = _next_layer;
	}


	// Feed function FOR THE FIRST LAYER ONLY. 
	void feed(MatrixXd input)
	{
		if(flags == FIRST_LAYER)
		{
			activation_vector = input.transpose() + bias_vector; // It seems you don't have to have a bias for the first layer but it doesn't hurt.
		}
		else
			std::cout << "Dont give input to layers besides the first.\n";
		
	}
	// feed function for all layers besides the first.
       	MatrixXd feed()
	{
		if(flags >= MID_LAYER)
		{
			// uses a tmp value so that each step of feedforward can be analyzed.
			MatrixXd tmp = weight_matrix.transpose();
			tmp = tmp * prev_layer->activation_vector;
			tmp = tmp + bias_vector;
			weighted_inputs = tmp;

			// can be replaced with:
			//weighted_inputs = ((weight_matrix.transpose() * prev_layer->activation_vector) + bias_vector);

			// unaryExpr takes a address of a function and applies it to all values in the 
			// matrix. This is assigning the activation vector, the values of the weighted_inputs
			// passed into the sigmoid function (sigm).
			activation_vector = weighted_inputs.unaryExpr(&sigm);

		}

		//return outputs from the layer. Can be ignored unless this is the last layer.
		return activation_vector;
	}


};

class Network
{
	// Network is currently similar to layers in that everything is public
	// bad programming practice. Should implement get methods, etc.
	public:
	RowVectorXL layers; // vector of layers of nodes (this runs the empty layers constructor.)
	int nol; // the number of layers in the network
	const double learning_const = 10; // Learning constant available for the network.

	Network(MatrixXi layer_sizes, int _nol)
	{
		nol = _nol;
		layers.resize(nol); //resize the layers array to be the size of the network.
		
		
		// initiliaze the first layer by assigning the previous_layer value to NULL
		// layer_sizes holds a array of each layer size (so 0 is the first layer.)
		layers(0).Layer_init(
				layer_sizes(0), 
				-1,			// pass -1 as the previous layer number of 
							// nodes since there is no previous layer 

				NULL, 			// NULL for pointer to previous layer.

				&layers(1));		// address of next layer

				
		// Roll through the network initializing each layer besides the first and last
		for(int i = 1; i < nol-1; i++)
		{
			layers(i).Layer_init(
					layer_sizes(i),
					layer_sizes(i-1),
					&layers(i-1),
					&layers(i+1));

		}
		
		// initialize the last layer.
		// note the the pointer to the next layer is set to NULL 
		layers(nol-1).Layer_init(
				layer_sizes(nol-1),
			       	layer_sizes(nol-2),
				&layers(nol-2),
				NULL);
	       	
	}

	// prints all the weights in the neural network.
	void printAllWeights()
	{
		std::cout << "----------------------------------------------------------------------\n\nWeights: \n";
		for(int i = 0; i < nol; i++)
		{
			std::cout << "layer" << i << ": \n";
			std::cout << layers(i).weight_matrix << "\n\n";
		}
		std::cout << "----------------------------------------------------------------------\n";
	}

	// prints out all the bias in the neural network
	void printAllBias()
	{
		std::cout << "----------------------------------------------------------------------\n\nBias: \n";
		for(int i = 0; i < nol; i++)
		{
			std::cout << "layer" << i << ": \n";
			std::cout << layers(i).bias_vector << "\n\n";
		}
		std::cout << "----------------------------------------------------------------------\n";
	}

	// Feedforward function
	// Calls the feed function of each layer in the neural network.
	// returns the results obtained from the last layer.
	MatrixXd feedforward(MatrixXd input)
	{
		
		layers(0).feed(input);
		for(int k = 1; k < nol-1; k++)
		{
			layers(k).feed();
		}
		MatrixXd result = layers(nol-1).feed();
		return result;
	}

	// The training algorithm for the neural network.
	// uses the backpropagation algorithm.
	//
	MatrixXd train(MatrixXd inputs, MatrixXd targets)
	{
		// first feedforward the network, to obtain a error value.
		VectorXd res = feedforward(inputs);

		// just a helpful shortcut (and helps with off-by-one errors.
		int last = nol-1;
		
		// backpropagation for the last layer
		// 

		// Calculate the derivative of the cost function for the last layer.
		MatrixXd tmp = layers(last).activation_vector - targets.transpose();

		if(DEBUG) std::cout << "act-targ:\n" << tmp << "\n";

		// calculate the sigmoid derivative of the last layers weighted inputs.
		MatrixXd tmp2 = layers(last).weighted_inputs.unaryExpr(&sigm_derivative);

		if(DEBUG) std::cout << "sigDer of weight_ins for last layer:\n" << tmp2 << "\n";

		// calculated delta for the last layer by multiplying coefficient wise
		// the derivative of the cost function at the last layer by the sigmoid deriv
		// of the last layers weighted inputs.
		tmp = tmp.cwiseProduct(tmp2);
		layers(last).delta = tmp; // assigns the above calculation to the last layers delta.


		if(DEBUG) std::cout << "delta: \n" << tmp << "\n";


		// calculate the derivative of the cost function with respect to each weight in the 
		// last layer. Uses the delta from above and the second to last layers
		// activation vector.
		layers(last).dC_dw = layers(last).delta * layers(last-1).activation_vector.transpose();

		if(DEBUG) std::cout << "act vec: \n" << layers(last-1).activation_vector << "\n";
		if(DEBUG) std::cout << "dC_dw:\n" << layers(last).dC_dw << "\n";


		// Backpropagation for middle layers starts with the second to last layer
		// and moves toward the first.
		for(int l = last-1; l > 0; l--)
		{
			// output what layer is being backproped.
			if(DEBUG) std::cout << "layer number: " << l << "\n\n";

			
			tmp = layers(l+1).weight_matrix * layers(l+1).delta;
			
			
			if(DEBUG) std::cout << "delta: \n" << layers(l+1).delta << "\n";
			if(DEBUG) std::cout << "weight matrix: \n" << layers(l+1).weight_matrix << "\n";
			
			// remember that unaryExpr evalues the function argument given
			// for each value in the matrix.
			tmp2 = layers(l).weighted_inputs.unaryExpr(&sigm_derivative);
			
			if(DEBUG) std::cout << "weighted_inputs: \n" << layers(l).weighted_inputs << "\n";
			if(DEBUG) std::cout << "sigm_derv of wei ins: \n" << tmp2 << "\n";

			// cwiseProduct is coefficient-wise product.
			layers(l).delta = tmp.cwiseProduct(tmp2);
			
			if(DEBUG) std::cout << "new delta: \n" << layers(l).delta << "\n";


			layers(l).dC_dw = layers(l).delta * layers(l-1).activation_vector.transpose();
			
			if(DEBUG) std::cout << "new dc_dw: \n" << layers(l).dC_dw << "\n";
		}


		// Here you can implement backprop for the bias vectors associated with the first 
		// layer of the network, but since the first layer is just a placeholder for the 
		// inputs given to the network, the algorithm is a little different.

		// Note that since the values entered into the network are not necessarily from 0 to 1
		// like the output from a sigmoid function will be, using the derivative of the sigmoid 
		// function may lead to strange results.
		
		// Also remember that the feed function for the first layer does not calculated a weighted
		// input. So any backprop method used to edit bias's for the first layer must take that 
		// into account.

		return res;

	}

	void updateWeights()
	{
		for(int l = 1; l < nol; l++)
		{
			//std::cout << "weights before: \n" << layers(l).weight_matrix << "\n";
			//std::cout << "dC_dw * lC: \n" << (layers(l).dC_dw.transpose() * learning_const) << "\n";
			layers(l).weight_matrix = layers(l).weight_matrix - (layers(l).dC_dw.transpose() * learning_const);
			//std::cout << "weights after: \n" << layers(l).weight_matrix << "\n";

			//std::cout << "bias_vec before: \n" << layers(l).bias_vector << "\n";
			//std::cout << "delta * lC: \n" << (layers(l).delta * learning_const) << "\n";
			layers(l).bias_vector = layers(l).bias_vector - (layers(l).delta * learning_const);
			//std::cout << "bias vec after: \n" << layers(l).bias_vector << "\n";
		}
	}
};


int main()
{

	// initialize the NN
	int nols = 3;

	RowVectorXi sizes(nols);

	sizes(0) = 3;
	sizes(1) = 6;
	sizes(2) = 3;

	Network neun(sizes, nols);

	// create the inputs
	MatrixXd inputs(8, 3);
	inputs << 0, 0, 0,
		  0, 0, 1,
		  0, 1, 0,
		  0, 1, 1,
		  1, 0, 0,
		  1, 0, 1,
		  1, 1, 0,
		  1, 1, 1;

	// Create the targets
	MatrixXd targets(8, 3);
	targets << 0, 0, 1,
		   0, 1, 0,
		   0, 1, 1,
		   1, 0, 0,
		   1, 0, 1,
		   1, 1, 0,
		   1, 1, 1,
		   0, 0, 0;
	
	MatrixXd results(3, 1);

	// give a shot at teaching it xor
	
	for(int i = 0; i < 500000; i++)
	{
		for(int x = 0; x < 8; x++)
		{
			neun.train(inputs.row(x), targets.row(x) );
			neun.updateWeights();	
		}

		if(i%10000 == 0)
		{
			neun.printAllWeights();
			for(int c = 0; c < 8; c++)
			{

				results = neun.feedforward(inputs.row(c));
				std::cout << "\t\tresults: \n" << results << "\n";
				pause();
			}
	
		}
	
	}

	neun.printAllWeights();
	for(int c = 0; c < 8; c++)
	{

		results = neun.feedforward(inputs.row(c));
		std::cout << "\t\tresults: \n" << results << "\n";
		pause();
	}


}
