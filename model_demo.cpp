#include <iostream>
#include <optional>

#include "sciplot/sciplot.hpp"
#include "autograd/autograd.h"

template <num::num_t T>
class RegressionModel : public nn::Module<T, RegressionModel<T>> {
public:
	RegressionModel()
	: linLayer1 (this->registerModule(nn::Linear<T>(2, 10))),
	  linLayer2 (this->registerModule(nn::Linear<T>(10, 1, false)))
	{}

	num::Tensor<T> forward(const num::Tensor<T>& x) const
	{
		num::Tensor<T> out = autofn::relu<T>(linLayer1.forward(x));
		return linLayer2.forward(out);
	}
private:
	nn::Linear<T> linLayer1;
	nn::Linear<T> linLayer2;
};

int main()
{
	RegressionModel<double> regModel;
	optim::Adam<double> opt(regModel.parameters, 0.1);

	num::Tensor<double> trainingData = num::randUniform<double>({1000, 2}, -5, 5);
	std::cout << "training data: " << trainingData.toString() << std::endl;
	
	num::Tensor<double> validationData = num::randUniform<double>({100, 2}, -5, 5);
	std::cout << "validation data: " << validationData.toString() << std::endl;

	std::vector<double> xV;
	std::vector<double> yV;
	std::vector<double> zV;
	std::vector<double> zPredV;

	sciplot::Plot3D goalPlot;
	sciplot::Plot3D outputPlotBeforeTrain;
	sciplot::Plot3D outputPlotAfterTrain;

    // Clear all borders and set the visible ones
    goalPlot.border().clear();
    goalPlot.border().bottomLeftFront();
    goalPlot.border().bottomRightFront();
    goalPlot.border().leftVertical();
	goalPlot.zrange(0.0, 90.0);

    outputPlotBeforeTrain.border().clear();
    outputPlotBeforeTrain.border().bottomLeftFront();
    outputPlotBeforeTrain.border().bottomRightFront();
   	outputPlotBeforeTrain.border().leftVertical();
	outputPlotBeforeTrain.zrange(0.0, 90.0);

    outputPlotAfterTrain.border().clear();
    outputPlotAfterTrain.border().bottomLeftFront();
    outputPlotAfterTrain.border().bottomRightFront();
   	outputPlotAfterTrain.border().leftVertical();
	outputPlotAfterTrain.zrange(0.0, 90.0);

	num::Tensor<double> modelInput({1,2});
	for (double x = -5; x < 5; x+=0.1) {
		for (double y = -5; y < 5; y+=0.1) {
			xV.push_back(x);
			yV.push_back(y);
			modelInput.setSingle(x, {0,0});
			modelInput.setSingle(y, {0,1});
			zV.push_back(std::pow(x,2) + std::pow(y,2));
			zPredV.push_back(regModel.forward(modelInput).getSingle({0,0}));
		}
	}

	goalPlot.drawDots(xV, yV, zV);
	outputPlotBeforeTrain.drawDots(xV, yV, zPredV);


	int batchSize = 50;
	int epochs = 30;


	for (int i = 0; i < epochs; i++) {

		// first way to loop over Tensor
		for (int j = 0; j < trainingData.dims[0]; j+=batchSize) {
			opt.zeroGradient();
			trainingData
				.get({num::Slice{j, j+batchSize, std::nullopt}})
				.iter([&regModel, &trainingData, &opt](const auto& idx, const auto& el) {
					num::Tensor<double> z = autofn::pow<double>(el.get({0,0}),2) +
							autofn::pow<double>(el.get({0,1}), 2);

					num::Tensor<double> zPred = regModel.forward(el);

					num::Tensor<double> loss = autofn::mseLoss<double>(zPred, z);
					loss.backward();
				});
			opt.step();

		}
		
		num::Tensor<double> valLossTotal(0);
		// another way to loop over Tensor
		for (int j = 0; j < validationData.dims[0]; j++) {
			num::Tensor<double> z = autofn::pow<double>(validationData.get({j,0}),2) +
					autofn::pow<double>(validationData.get({j,1}), 2);

			num::Tensor<double> zPred = regModel.forward(validationData.get({j}));

			valLossTotal = valLossTotal + autofn::mseLoss<double>(zPred, z);
		}
		valLossTotal = valLossTotal / num::Tensor<double>(validationData.dims[0]);
		std::cout << "Epoch " << i << " average validation loss: " << valLossTotal.get({0}).toString() << std::endl;
	}

	zPredV.clear();
	std::cout << "before computing prediction" << std::endl;
	for (double x = -5; x < 5; x+=0.1) {
		for (double y = -5; y < 5; y+=0.1) {
			modelInput.setSingle(x, {0,0});
			modelInput.setSingle(y, {0,1});
			zPredV.push_back(regModel.forward(modelInput).getSingle({0,0}));
		}
	}

	std::cout << "Plotting..." << std::endl;

	outputPlotAfterTrain.drawDots(xV, yV, zPredV);

	sciplot::Figure fig = {{goalPlot, outputPlotBeforeTrain, outputPlotAfterTrain}};
	sciplot::Canvas canvas = {{fig}};
	canvas.size(1920,1080);
	canvas.save("plot.png");
}