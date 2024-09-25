#pragma once
#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <map>
#include <memory>
#include <set>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace opennn;

struct TrainingParams
{
	TrainingStrategy::OptimizationMethod optMeth;
	TrainingStrategy::LossMethod lossMeth;
};

struct SingleResult
{
	TrainingStrategy::OptimizationMethod optMeth;
	TrainingStrategy::LossMethod lossMeth;
	int selected;
	int hits;
	int points;
};

struct SingleTestResult
{
	std::string testName;
	int selected;
	int hits;
	int points;
	std::vector<float> results;
	std::vector<int> expected;
};

struct SumOfTests_1
{
	std::string name;
	float sum_average_1;
	float sum_avgPoints_1;
	float sum_average_3;
	float sum_avgPoints_3;
	float sum_average_6;
	float sum_avgPoints_6;
	float sum_average_12;
	float sum_avgPoints_12;

	bool operator<(const SumOfTests_1& rhs) const { return (name < rhs.name); }
	bool operator==(const SumOfTests_1& rhs) const { return (name == rhs.name); }
};

struct SumOfTests
{
	std::string name;
	float average_1;
	float avgPoints_1;
	float average_3;
	float avgPoints_3;
	float average_6;
	float avgPoints_6;
	float average_12;
	float avgPoints_12;

	bool operator< (const SumOfTests& other) const
	{
		if (average_1 == other.average_1)
		{
			if (avgPoints_1 == other.avgPoints_1)
			{
				if (average_3 == other.average_3)
				{
					if (avgPoints_3 == other.avgPoints_3)
					{
						if (average_6 == other.average_6)
						{
							if (avgPoints_6 == other.avgPoints_6)
							{
								if (average_12 == other.average_12)
								{
									return avgPoints_12 > other.avgPoints_12;
								}
								return average_12 > other.average_12;
							}
							return avgPoints_6 > other.avgPoints_6;
						}
						return average_6 > other.average_6;
					}
					return avgPoints_3 > other.avgPoints_3;
				}
				return average_3 > other.average_3;
			}
			return avgPoints_1 > other.avgPoints_1;
		}
		else
		{
			return average_1 > other.average_1;
		}
	}

	bool operator()(const SumOfTests& other) const
	{
		if (average_1 == other.average_1)
		{
			if (avgPoints_1 == other.avgPoints_1)
			{
				if (average_3 == other.average_3)
				{
					if (avgPoints_3 == other.avgPoints_3)
					{
						if (average_6 == other.average_6)
						{
							if (avgPoints_6 == other.avgPoints_6)
							{
								if (average_12 == other.average_12)
								{
									return avgPoints_12 > other.avgPoints_12;
								}
								return average_12 > other.average_12;
							}
							return avgPoints_6 > other.avgPoints_6;
						}
						return average_6 > other.average_6;
					}
					return avgPoints_3 > other.avgPoints_3;
				}
				return average_3 > other.average_3;
			}
			return avgPoints_1 > other.avgPoints_1;
		}
		else
		{
			return average_1 > other.average_1;
		}
	}
};

class Analizer
{
public:
	Analizer(const std::string& rootFolder);

	void SetTrainigParams(const Index& hiddenNeuronsNumber, const float& selectionPoint, const int& number_of_repeates, const float& acceptModelConditionParam, const int& topHitLevel);
	void StartTraining(const std::string& locationFolder, const std::string& tryPostFix);
	void StartTrainingWithVecOfParams(const std::string& locationFolder, const std::string& tryPostFix, const std::vector<TrainingParams>& vecOfParams);
	void TestSingleModel(const std::string& testDataFolder, const std::string& modelsFolder, const std::string& modelName);
	void TestDataWithModels(const std::string& testDataFolder, const std::string& modelsFolder, const std::vector<std::string>& modelsNames);
	void GetTestResultsForNewDraw(const std::string& testDataFolder, const std::string& modelsFolder, const std::vector<std::string>& modelsNames);
	void TestGivenModelsWithDatas(const std::vector<std::string>& containerOfDatas, const std::string& modelsFolder);

private:
	std::string m_rootFolder;
	DataSet m_dataSet;
	Index m_inputVariablesNumber{ 0 };
	Index m_targetVariablesNumber{ 0 };
	Index m_hiddenNeuronsNumber;

	std::vector<std::initializer_list<type>> m_tempTestDataVoI;
	std::vector<std::vector<type>> m_tempTestDataVoV;
	std::initializer_list<std::initializer_list<type>> m_testData;

	std::vector<int> m_expectedResults;
	float m_selection_point;
	int m_number_of_repeates;
	float m_acceptModelConditionParam;
	int m_topHitLevel;

	std::vector<TrainingStrategy::OptimizationMethod> m_optimizationMethods;
	std::vector<TrainingStrategy::LossMethod> m_lossMethods;

	std::shared_ptr<NeuralNetwork> GetTrainedNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Index& hiddenNeuronsNumber, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod);

	std::shared_ptr<NeuralNetwork> LoadNeuralNetwork(const std::string& fileName);

	Tensor<type, 2> TestWithData(std::shared_ptr<NeuralNetwork> neuralNetwork);

	std::vector<SingleResult> CollectResult(std::shared_ptr<NeuralNetwork> neuralNetwork, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod);

	void LoadTestInputData(const std::string& givenFolderName);
	void LoadTestExpectedResults(const std::string& givenFolderName);

	std::string GetTestFileName(const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod);

	bool AcceptNNModelCondition(const int& selectedNumbers, const int& hitedNumbers, const int& sumOfPoints, const int& topNumbers, const int& hitedFromTop, const int& pointsFromTop);

	std::string GetTestFolderName(const std::string& testDataFolder, const std::string& modelsFolder);
	std::string GetExcelFileName(const std::string& testDataFolder, const std::string& modelsFolder);
	void CreateTestFolder(const std::string& testFolder);
	void CreateModelsFolder(const std::string& folderName);
	opennn::type GetBorderTopValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	opennn::type GetBorder3sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	opennn::type GetBorder6sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
	opennn::type GetBorder12sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults);
};
