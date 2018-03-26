#define FIRST_LAYER 1
#define MID_LAYER 2
#define LAST_LAYER 3
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class Layer;

typedef Matrix<Layer, 1, Dynamic> RowVectorXL;

void pause()
{
	std::cout << "Press enter to continue ...";
	std::cin.get();
}

double sigm(double x)
{
	return 1 / (1 + exp(-1*x));
}
double sigm_derivative(double x)
{
	return sigm(x) * (1 - sigm(x));
}


class Layer
{
	public:
	int flags;
	MatrixXd weight_matrix;
	int number_of_nodes;
	VectorXd bias_vector;
	VectorXd activation_vector;
	VectorXd weighted_inputs;
	Layer* next_layer;
	Layer* prev_layer;
	VectorXd delta;
	MatrixXd dC_dw;
	const double L_C = 1000;

	Layer()
	{

	}


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


	void 
	Layer_init(int _number_of_nodes, 
			int prev_layer_number_of_nodes, 
			Layer* _prev_layer, 
			Layer* _next_layer)
	{
		if(_prev_layer == NULL)
		{
		
			flags = FIRST_LAYER;
		} else if(_next_layer == NULL)
		{

			flags = LAST_LAYER;
		} else {

			flags = MID_LAYER;
		}
		
		number_of_nodes = _number_of_nodes;
		if(flags != FIRST_LAYER)
		{
			weight_matrix = MatrixXd::Random(prev_layer_number_of_nodes, number_of_nodes);
		}
		delta.resize(number_of_nodes, 1);
		bias_vector = MatrixXd::Random(number_of_nodes, 1);
		activation_vector.resize(number_of_nodes, 1);
		weighted_inputs.resize(number_of_nodes, 1);
		if(flags != FIRST_LAYER)	
			prev_layer = _prev_layer;
		if(flags != LAST_LAYER)
			next_layer = _next_layer;
	}

	void feed(MatrixXd input)
	{
		if(flags == FIRST_LAYER)
		{
			activation_vector = input.transpose() + bias_vector;
		}
		else
			std::cout << "Dont give input to layers besides the first.\n";
		
	}
       	MatrixXd feed()
	{
		if(flags >= MID_LAYER)
		{
		
			MatrixXd tmp = weight_matrix.transpose();
			tmp = tmp * prev_layer->activation_vector;
			tmp = tmp + bias_vector;
			weighted_inputs = tmp;

			//weighted_input = ((weight_matrix.transpose() * prev_layer->activation_vector) + bias_vector);

			activation_vector = weighted_inputs.unaryExpr(&sigm);

		}
		return activation_vector;
	}


};

class Network
{
	public:
	RowVectorXL layers;
	int nol;
	const double learning_const = 10;

	Network(MatrixXi layer_sizes, int _nol)
	{
		nol = _nol;
		layers.resize(nol);
		
		
		layers(0).Layer_init(
				layer_sizes(0), 
				-1, 
				NULL, 
				&layers(1));

				
		for(int i = 1; i < nol-1; i++)
		{
			layers(i).Layer_init(
					layer_sizes(i),
					layer_sizes(i-1),
					&layers(i-1),
					&layers(i+1));

		}
		
		
		layers(nol-1).Layer_init(
				layer_sizes(nol-1),
			       	layer_sizes(nol-2),
				&layers(nol-2),
				NULL);
	       	
	}

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

	MatrixXd train(MatrixXd inputs, MatrixXd targets)
	{
		VectorXd res = feedforward(inputs);

		int last = nol-1;
		
		// backprop for the last layer
		MatrixXd tmp = layers(last).activation_vector - targets;
		//std::cout << "act-targ:\n" << tmp << "\n";
		MatrixXd tmp2 = layers(last).weighted_inputs.unaryExpr(&sigm_derivative);
		//std::cout << "sigDer of weight_ins:\n" << tmp2 << "\n";

		tmp = tmp.cwiseProduct(tmp2);
		layers(last).delta = tmp;
		//std::cout << "delta: \n" << tmp << "\n";

		layers(last).dC_dw = layers(last).delta * layers(last-1).activation_vector.transpose();
		//std::cout << "act vec: \n" << layers(last-1).activation_vector << "\n";
		//std::cout << "dC_dw:\n" << layers(last).dC_dw << "\n";

		for(int l = last-1; l > 0; l--)
		{

			//std::cout << "l: " << l << "\n\n";
			tmp = layers(l+1).weight_matrix * layers(l+1).delta;
			//std::cout << "delta: \n" << layers(l+1).delta << "\n";
			//std::cout << "weight matrix: \n" << layers(l+1).weight_matrix << "\n";
			tmp2 = layers(l).weighted_inputs.unaryExpr(&sigm_derivative);
			//std::cout << "weighted_inputs: \n" << layers(l).weighted_inputs << "\n";
			//std::cout << "sigm_derv of wei ins: \n" << tmp2 << "\n";

			layers(l).delta = tmp.cwiseProduct(tmp2);
			//std::cout << "new delta: \n" << layers(l).delta << "\n";

			layers(l).dC_dw = layers(l).delta * layers(l-1).activation_vector.transpose();
			//std::cout << "new dc_dw: \n" << layers(l).dC_dw << "\n";
		}


		// really don't need this, I can just make initial bias 0?
		//tmp = layers(1).weight_matrix.transpose() * layers(1).delta;
		//tmp2 = layers(0).activation_vector.unaryExpr(&sigm_derivative);

		//layers(0).delta = tmp.cwiseProduct(tmp2);

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
	sizes(1) = 3;
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
			neun.train(inputs.row(x), targets.row(x).transpose());
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
