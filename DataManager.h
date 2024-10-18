#pragma once

#include "Helper.h"

class DataManager
{
public:
	DataManager(const std::string& rootFolder);

	void StartTraining(const MODEL_DESTINATION modelDestination, const std::string& inputDataFolder, const TrainingParameters& trainingParameters);
private:
	std::string m_rootFolder;
	DataSet m_dataSet;
	std::vector<std::vector<type>> m_tempTestDataVector;
	std::vector<std::initializer_list<type>> m_testDataVector;
	std::initializer_list<std::initializer_list<type>> m_testData;
	std::array<float, 4U> m_limitValues;

	std::vector<std::string> GetAllNamesFromLocation(const std::string& location);
	void CreateFolder(const std::string& path);
	std::string GetLocationForModels(const std::string& inputDataLocation, int& orderModelsFolderNumber, const MODEL_DESTINATION modelDestination);
	std::string GetInputDataFileName(const MODEL_DESTINATION modelDestination, const std::string& inputDataFolder);
	std::string GetModelDestinantionName(const MODEL_DESTINATION modelDestination);
	void LoadTestInputData(const std::string& inputDataLocation);
	std::vector<int> LoadTestExpectedResults(const MODEL_DESTINATION modelDestination, const std::string& inputDataLocation);
	std::string GetShortsOfAlgorithms(const NeuralNetwork::ProjectType& projectType, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod);
	std::shared_ptr<NeuralNetwork> GetTrainedNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Index& hiddenNeuronsNumber, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod, const Index& inputVariablesNumber, const Index& targetVariablesNumber);
	SingleModelResult CollectResult(std::shared_ptr<NeuralNetwork> neuralNetwork, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod, const std::vector<int>& expectedResults);
	Tensor<type, 2> TestWithData(std::shared_ptr<NeuralNetwork> neuralNetwork);
	opennn::type GetBorderTopValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	opennn::type GetBorder3sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	opennn::type GetBorder6sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	opennn::type GetBorder12sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	bool IsFulfillExpectations(const SingleModelResult& results, const TrainingExpectations& expectations);
	void SaveTrainingParameters(const MODEL_DESTINATION modelDestination, const TrainingParameters& trainingParameters, const std::string& saveFolder);
	void RecalculateModelTestStatsAndStore(const std::string& modelLocation, const std::string& modelName, const std::vector<float>& data, const int& numberOfRounds, std::set<TempSumOfModelTestStats>& buffContainer, const std::vector<int>& expectedResults, const opennn::type& border_for_top_value, const opennn::type& border_for_3s, const opennn::type& border_for_6s, const opennn::type& border_for_12s);
};
