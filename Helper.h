#pragma once
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "includes/opennn/opennn.h"

using namespace opennn;

const std::string MODEL_FOLDER_NAME = "models";

enum class MODEL_DESTINATION
{
	MD_LOTTERY,
	MD_CANCER_DATA,
	MD_DUMMY_TEST
};

enum class ALGORITHMS_BUNCH
{
	AB_ALL,
	AB_ALL_WORST,
	AB_ALL_1_BEST,
	AB_ALL_3_BEST,
	AB_ALL_6_BEST,
	AB_ALL_12_BEST,
};

struct BestValues
{
	float at_1s;
	float at_3s;
	float at_6s;
	float at_12s;
};

struct BunachOdBests
{
	BestValues lottery_b;
	BestValues cancer_data_b;
	BestValues dummy_b;
};

struct ModelLocationPair
{
	std::string location;
	std::string modelName;

	bool operator< (const ModelLocationPair& other) const
	{
		return (location + modelName) < (other.location + other.modelName);
	}
};

struct NumberValuePoints
{
	int number;
	float value;
	float points;
};

struct ModelsNamesStruct
{
	MODEL_DESTINATION type;
	std::vector<ModelLocationPair> names;
};

struct TrainingExpectations
{
	float efficiency_only_top;
	float points_only_top;
	float efficiency_of_1_high;
	float efficiency_of_any_1;
	float efficiency_any_3;
	//float points3sLevel;
	float efficiency_any_6;
	//float points6sLevel;
	float efficiency_any_12;
	//float points12sLevel;
};

struct TrainingParameters
{
	Index hiddenNeuronsNumber;
	int numberOfReps;
	std::vector<NeuralNetwork::ProjectType> projectTypes;
	std::vector<TrainingStrategy::OptimizationMethod> optimizationMethods;
	std::vector<TrainingStrategy::LossMethod> lossMethods;
	TrainingExpectations expectations;
};

struct SingleTestResultPair
{
	std::string testName;
	std::vector<float> results;
};

struct SingleModelResult
{
	TrainingStrategy::OptimizationMethod optMeth;
	TrainingStrategy::LossMethod lossMeth;
	float efficiency_only_top{ 0.0f };
	float points_only_top{ 0.0f };
	float efficiency_of_1_high{ 0.0f };
	float efficiency_of_any_1{ 0.0f };
	float efficiency_any_3{ 0.0f };
	//float top3sPoints{ 0.0f };
	float efficiency_any_6{ 0.0f };
	//float top6sPoints{ 0.0f };
	float efficiency_any_12{ 0.0f };
	//float top12sPoints{ 0.0f };
};

struct ModelTestStats
{
	int only_top_select{ 0 };
	int only_top_hit{ 0 };
	int only_top_points{ 0 };

	int high_1_select{ 0 };
	int high_1_hit{ 0 };
	int high_1_points{ 0 };

	int any_1_select{ 0 };
	int any_1_hit{ 0 };
	int any_1_points{ 0 };

	int select_3s{ 0 };
	int hit_3s{ 0 };
	/*int points_3s{ 0 };*/

	int select_6s{ 0 };
	int hit_6s{ 0 };
	/*int points_6s{ 0 };*/

	int select_12s{ 0 };
	int hit_12s{ 0 };
	/*int points_12s{ 0 };*/
};

struct TempSumOfModelTestStats
{
	std::string location;
	std::string name;
	float sum_average_top_1;
	float sum_avgPoints_top_1;
	float sum_average_high_1;
	float sum_average_any_1;
	float sum_average_3;
	//float sum_avgPoints_3;
	float sum_average_6;
	//float sum_avgPoints_6;
	float sum_average_12;
	//float sum_avgPoints_12;

	bool operator<(const TempSumOfModelTestStats& rhs) const { return ((location + name) < (rhs.location + rhs.name)); }
	bool operator==(const TempSumOfModelTestStats& rhs) const { return ((location + name) == (rhs.location + rhs.name)); }
};

struct ModelTestResultAgregat
{
	std::string location;
	std::string name;
	float average_top_1;
	float avgPoints_top_1;
	float average_high_1;
	float average_any_1;
	float average_3;
	//float avgPoints_3;
	float average_6;
	//float avgPoints_6;
	float average_12;
	//float avgPoints_12;

	bool operator< (const ModelTestResultAgregat& other) const
	{
		if (average_top_1 == other.average_top_1)
		{
			if (average_high_1 == other.average_high_1)
			{
				if (average_any_1 == other.average_any_1)
				{
					if (avgPoints_top_1 == other.avgPoints_top_1)
					{
						if (average_3 == other.average_3)
						{
							if (average_6 == other.average_6)
							{
								if (average_12 == other.average_12)
								{
									return (location + name) > (other.location + other.name);
								}
								return average_12 > other.average_12;
							}
							return average_6 > other.average_6;
						}
						return average_3 > other.average_3;
					}
					return avgPoints_top_1 > other.avgPoints_top_1;
				}
				return average_any_1 > other.average_any_1;
			}
			return average_high_1 > other.average_high_1;
		}
		else
		{
			return average_top_1 > other.average_top_1;
		}
	}

	bool operator()(const ModelTestResultAgregat& other) const
	{
		if (average_top_1 == other.average_top_1)
		{
			if (average_high_1 == other.average_high_1)
			{
				if (average_any_1 == other.average_any_1)
				{
					if (avgPoints_top_1 == other.avgPoints_top_1)
					{
						if (average_3 == other.average_3)
						{
							if (average_6 == other.average_6)
							{
								if (average_12 == other.average_12)
								{
									return (location + name) > (other.location + other.name);
								}
								return average_12 > other.average_12;
							}
							return average_6 > other.average_6;
						}
						return average_3 > other.average_3;
					}
					return avgPoints_top_1 > other.avgPoints_top_1;
				}
				return average_any_1 > other.average_any_1;
			}
			return average_high_1 > other.average_high_1;
		}
		else
		{
			return average_top_1 > other.average_top_1;
		}
	}
};
