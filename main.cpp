#include "DataManager.h"

int main()
{
	DataManager dm("data");

	dm.StartTraining(MODEL_DESTINATION::MD_CANCER_DATA, "test_004",
		{
			.hiddenNeuronsNumber = 400,
			.numberOfReps = 3,
			.projectTypes =
			{
			NeuralNetwork::ProjectType::Classification,
			NeuralNetwork::ProjectType::Approximation,
			NeuralNetwork::ProjectType::AutoAssociation,
			NeuralNetwork::ProjectType::Forecasting
		},
		.optimizationMethods =
		{
			TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT,
			TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT,
			TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD,
			/*TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM,*/
			TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT,
			TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION
		},
		.lossMethods =
		{
			TrainingStrategy::LossMethod::SUM_SQUARED_ERROR,
			TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR,
			TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR,
			/*TrainingStrategy::LossMethod::MINKOWSKI_ERROR,*/
			TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR,
			TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR
		},
		.expectations =
			{
				.points_only_top = 0.9f,
				.efficiency_of_1_high = 0.0f,
				.efficiency_of_any_1 = 0.0f,
				.efficiency_any_3 = 0.0f,
				.efficiency_any_6 = 0.0f,
				.efficiency_any_12 = 0.0f
			}
		}
	);
}
