#include "DataManager.h"

//----------------------------------------------------------
DataManager::DataManager(const std::string& rootFolder)
	:m_rootFolder(rootFolder + "/")
{
	m_limitValues = { 0.95, 0.85, 0.75, 0.65 };
}
//----------------------------------------------------------
void DataManager::StartTraining(const MODEL_DESTINATION modelDestination, const std::string& inputDataFolder, const TrainingParameters& trainingParameters)
{
	std::string input_data_location = m_rootFolder + inputDataFolder;

	int order_models_folder_number = 1;
	auto folder_for_models_name = GetLocationForModels(input_data_location, order_models_folder_number, modelDestination);
	SaveTrainingParameters(modelDestination, trainingParameters, folder_for_models_name);

	auto input_data_file_name = GetInputDataFileName(modelDestination, input_data_location);

	if (input_data_file_name.empty())
	{
		std::cout << "[ERROR] Empty input data file name !" << std::endl;
		return;
	}

	const std::string file_sufix = GetModelDestinantionName(modelDestination) + "_" + inputDataFolder + "_" + std::to_string(trainingParameters.hiddenNeuronsNumber);

	m_dataSet.set(input_data_file_name, ';', true);

	Index input_variables_number = m_dataSet.get_input_variables_number();
	Index target_variables_number = m_dataSet.get_target_variables_number();

	LoadTestInputData(input_data_location);

	auto expected_results = LoadTestExpectedResults(modelDestination, input_data_location);

	//std::map<std::string, SingleModelResult> tests_summarize;

	for (const auto& project_type : trainingParameters.projectTypes)
	{
		for (const auto& optimization_method : trainingParameters.optimizationMethods)
		{
			for (const auto& loss_method : trainingParameters.lossMethods)
			{
				if (project_type == opennn::NeuralNetwork::ProjectType::Approximation ||
					project_type == opennn::NeuralNetwork::ProjectType::AutoAssociation ||
					project_type == opennn::NeuralNetwork::ProjectType::Forecasting)
				{
					if (optimization_method == opennn::TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT ||
						optimization_method == opennn::TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT ||
						optimization_method == opennn::TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD ||
						optimization_method == opennn::TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT ||
						optimization_method == opennn::TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
					{
						if (loss_method == opennn::TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR ||
							loss_method == opennn::TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR)
						{
							continue;
						}
					}
					if (optimization_method == opennn::TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
					{
						if (loss_method == opennn::TrainingStrategy::LossMethod::SUM_SQUARED_ERROR)
						{
							continue;
						}
					}
				}

				int repeate_counter = 0;
				const std::string shorts_combination_name = GetShortsOfAlgorithms(project_type, optimization_method, loss_method);

				while (repeate_counter < trainingParameters.numberOfReps)
				{
					auto neural_network = GetTrainedNeuralNetwork(project_type, trainingParameters.hiddenNeuronsNumber, optimization_method, loss_method, input_variables_number, target_variables_number);

					if (neural_network)
					{
						auto test_result = CollectResult(neural_network, optimization_method, loss_method, expected_results);

						if (IsFulfillExpectations(test_result, trainingParameters.expectations))
						{
							repeate_counter++;
							std::string try_name = shorts_combination_name + "_" + std::to_string(order_models_folder_number) + "_" + std::to_string(repeate_counter);

							//tests_summarize[try_name] = test_result;

							std::string file_name = folder_for_models_name + "/ " + file_sufix + "_" + try_name + ".xml";

							neural_network->save(file_name);
						}
					}
				}
			}
		}
	}

	int o = 0;
}
//----------------------------------------------------------
std::vector<std::string> DataManager::GetAllNamesFromLocation(const std::string& location)
{
	std::vector<std::string> response;
	for (const auto& entry : fs::directory_iterator(location))
	{
		// Converting the path to const char * in the
		// subsequent lines
		std::filesystem::path outfilename = entry.path();
		std::string outfilename_str = outfilename.string();
		std::size_t found = outfilename_str.find("\\");
		if (found != std::string::npos)
		{
			response.push_back(std::string(outfilename_str.begin() + (found + 1), outfilename_str.end()));
		}
	}

	return response;
}
//----------------------------------------------------------
void DataManager::CreateFolder(const std::string& path)
{
	fs::create_directory(path);

	if (fs::exists(path))
	{
		std::cout << "[INFO] Directory " << path << " created successfully." << std::endl;
	}
	else
	{
		std::cout << "[ERROR] Failed to create directory " << path << std::endl;
	}
}
//----------------------------------------------------------
std::string DataManager::GetLocationForModels(const std::string& inputDataLocation, int& orderModelsFolderNumber, const MODEL_DESTINATION modelDestination)
{
	auto potential_model_folders = GetAllNamesFromLocation(inputDataLocation);

	std::vector<int> already_used_numbers;
	for (auto& el : potential_model_folders)
	{
		if (el.find(MODEL_FOLDER_NAME) != std::string::npos)
		{
			std::size_t found = el.find("_");
			if (found != std::string::npos)
			{
				already_used_numbers.push_back(std::stoi(std::string(el.begin() + (found + 1), el.end())));
			}
			else
			{
				std::cout << "[ERROR] Wrong folder name: " << el << std::endl;
			}
		}
	}

	if (!already_used_numbers.empty())
	{
		std::string last_created_location = inputDataLocation + "/" + MODEL_FOLDER_NAME + "_" + std::to_string(already_used_numbers.back());

		auto files_in_last_created_location = GetAllNamesFromLocation(last_created_location);

		if (files_in_last_created_location.empty())
		{
			return last_created_location;
		}
		else
		{
			std::vector<std::string> vec_of_configuration_files;
			for (const auto& f_name : files_in_last_created_location)
			{
				if (f_name.find("configuration") != std::string::npos)
				{
					vec_of_configuration_files.push_back(f_name);
				}
			}

			bool has_already_this_type = false;
			for (const auto& conf_file : vec_of_configuration_files)
			{
				if (files_in_last_created_location.at(0).find(GetModelDestinantionName(modelDestination)) != std::string::npos)
				{
					has_already_this_type = true;
					break;
				}
			}

			if (!has_already_this_type)
			{
				return last_created_location;
			}
		}

		orderModelsFolderNumber += already_used_numbers.back();
	}

	std::string new_location = inputDataLocation + "/" + MODEL_FOLDER_NAME + "_" + std::to_string(orderModelsFolderNumber);

	CreateFolder(new_location);

	return new_location;
}
//----------------------------------------------------------
std::string DataManager::GetInputDataFileName(const MODEL_DESTINATION modelDestination, const std::string& inputDataFolder)
{
	switch (modelDestination)
	{
	case MODEL_DESTINATION::MD_LOTTERY:
	{
		return inputDataFolder + "/lottery_input_data.csv";
	}
	case MODEL_DESTINATION::MD_CANCER_DATA:
	{
		return inputDataFolder + "/cancer_data_input_data.csv";
	}
	case MODEL_DESTINATION::MD_DUMMY_TEST:
	{
		return inputDataFolder + "/dummy_input_data.csv";
	}
	default:
	{
		std::cout << "[ERROR] Unknown MODEL_DESTINATION type !" << std::endl;
		break;
	}
	}

	return "";
}
//----------------------------------------------------------
std::string DataManager::GetModelDestinantionName(const MODEL_DESTINATION modelDestination)
{
	switch (modelDestination)
	{
	case MODEL_DESTINATION::MD_LOTTERY:
	{
		return "LOTTERY";
	}
	case MODEL_DESTINATION::MD_CANCER_DATA:
	{
		return "CANCER_DATA";
	}
	case MODEL_DESTINATION::MD_DUMMY_TEST:
	{
		return "DUMMY_TEST";
	}
	default:
	{
		std::cout << "[ERROR] Unknown MODEL_DESTINATION type !" << std::endl;
		break;
	}
	}

	return "UNKNOWN";
}
//----------------------------------------------------------
void DataManager::LoadTestInputData(const std::string& inputDataLocation)
{
	m_tempTestDataVector.clear();
	m_testDataVector.clear();

	const std::string file_name = inputDataLocation + "/test_input_data.txt";
	std::ifstream myfile(file_name);
	std::string mystring;

	int buff_index = 0;
	if (myfile.is_open())
	{
		while (myfile.good())
		{
			myfile >> mystring;

			m_tempTestDataVector.push_back(std::vector<type>());
			std::stringstream ss(mystring);

			while (ss.good())
			{
				std::string substr;
				std::getline(ss, substr, ';');
				m_tempTestDataVector.at(buff_index).push_back(type(std::stof(substr)));
			}

			m_testDataVector.push_back(std::initializer_list<type>(m_tempTestDataVector.at(buff_index).data(), m_tempTestDataVector.at(buff_index).data() + m_tempTestDataVector.at(buff_index).size()));

			++buff_index;
		}
	}
	else
	{
		std::cout << "[ERROR] Cannot open file " + file_name << std::endl;
	}

	m_testData = std::initializer_list<std::initializer_list<type>>(m_testDataVector.data(), m_testDataVector.data() + m_testDataVector.size());
}
//----------------------------------------------------------
std::vector<int> DataManager::LoadTestExpectedResults(const MODEL_DESTINATION modelDestination, const std::string& inputDataLocation)
{
	std::vector<int> response;

	std::string file_name = inputDataLocation;

	switch (modelDestination)
	{
	case MODEL_DESTINATION::MD_LOTTERY:
	{
		file_name += "/lottery_expected_results.txt";
		break;
	}
	case MODEL_DESTINATION::MD_CANCER_DATA:
	{
		file_name += "/cancer_data_expected_results.txt";
		break;
	}
	case MODEL_DESTINATION::MD_DUMMY_TEST:
	{
		file_name += "/dummy_expected_results.txt";
		break;
	}
	default:
	{
		std::cout << "[ERROR] Unknown MODEL_DESTINATION type !" << std::endl;
		break;
	}
	}

	std::ifstream myfile(file_name);
	std::string mystring;

	if (myfile.is_open())
	{
		while (myfile.good())
		{
			myfile >> mystring;

			if (!mystring.empty())
			{
				response.push_back(std::stoi(mystring));
			}
		}
	}
	else
	{
		std::cout << "[ERROR] Cannot open file " + file_name << std::endl;
	}

	return response;
}
//----------------------------------------------------------
std::string DataManager::GetShortsOfAlgorithms(const NeuralNetwork::ProjectType& projectType, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod)
{
	std::string response = "";

	switch (projectType)
	{
	case NeuralNetwork::ProjectType::Approximation:
	{
		response = "AX_";
		break;
	}
	case NeuralNetwork::ProjectType::Classification:
	{
		response = "CL_";
		break;
	}
	case NeuralNetwork::ProjectType::Forecasting:
	{
		response = "FR_";
		break;
	}
	case NeuralNetwork::ProjectType::ImageClassification:
	{
		response = "IC_";
		break;
	}
	case NeuralNetwork::ProjectType::TextClassification:
	{
		response = "TC_";
		break;
	}
	case NeuralNetwork::ProjectType::TextGeneration:
	{
		response = "TG_";
		break;
	}
	case NeuralNetwork::ProjectType::AutoAssociation:
	{
		response = "AA_";
		break;
	}
	default:
		break;
	}

	switch (optimizationMethod)
	{
	case TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT:
	{
		response += "GD_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT:
	{
		response += "CG_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD:
	{
		response += "QN_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM:
	{
		response += "LM_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT:
	{
		response += "SG_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION:
	{
		response += "AM_";
		break;
	}
	default:
		break;
	}

	switch (lossMethod)
	{
	case TrainingStrategy::LossMethod::SUM_SQUARED_ERROR:
	{
		response += "SS";
		break;
	}
	case TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR:
	{
		response += "MS";
		break;
	}
	case TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR:
	{
		response += "NS";
		break;
	}
	case TrainingStrategy::LossMethod::MINKOWSKI_ERROR:
	{
		response += "ME";
		break;
	}
	case TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR:
	{
		response += "WQ";
		break;
	}
	case TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR:
	{
		response += "CE";
		break;
	}
	default:
		break;
	}

	return response;
}
//----------------------------------------------------------
std::shared_ptr<NeuralNetwork> DataManager::GetTrainedNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Index& hiddenNeuronsNumber, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod, const Index& inputVariablesNumber, const Index& targetVariablesNumber)
{
	try
	{
		srand(static_cast<unsigned>(time(nullptr)));

		std::shared_ptr<NeuralNetwork> neural_network(new NeuralNetwork(projectType, { inputVariablesNumber, hiddenNeuronsNumber, targetVariablesNumber }));

		TrainingStrategy training_strategy(neural_network.get(), &m_dataSet);

		training_strategy.set_loss_method(lossMethod);
		training_strategy.set_optimization_method(optimizationMethod);
		training_strategy.perform_training();

		return neural_network;
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;

		return nullptr;
	}
}
//----------------------------------------------------------
SingleModelResult DataManager::CollectResult(std::shared_ptr<NeuralNetwork> neuralNetwork, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod, const std::vector<int>& expectedResults)
{
	try
	{
		auto test_results = TestWithData(neuralNetwork);

		std::vector<float> single_results;

		for (int i = 0; i < test_results.dimension(0); ++i)
		{
			for (int j = 0; j < test_results.dimension(1); ++j)
			{
				single_results.push_back(test_results(i, j));
			}
		}

		std::multiset<ModelTestResultAgregat> l_results_agregats;
		std::set<TempSumOfModelTestStats> buff_container;

		int amount_of_draws_in_data = static_cast<int>(single_results.size() / 49);

		for (int round = 0; round < amount_of_draws_in_data; ++round)
		{
			auto begin_itr = single_results.begin() + (round * 49);
			std::vector<float> single_round_results(begin_itr, begin_itr + 49);

			auto lottery_expect_begin_itr = expectedResults.begin() + (round * 49);
			std::vector<int> round_lottery_expect(lottery_expect_begin_itr, lottery_expect_begin_itr + 49);

			std::set<float> buff_set_of_results(single_round_results.begin(), single_round_results.end());
			std::vector<float> ordered_results(buff_set_of_results.begin(), buff_set_of_results.end());
			std::reverse(std::begin(ordered_results), std::end(ordered_results));

			opennn::type border_for_top_value = GetBorderTopValue(single_round_results, ordered_results);
			opennn::type border_for_3s = GetBorder3sValue(single_round_results, ordered_results);
			opennn::type border_for_6s = GetBorder6sValue(single_round_results, ordered_results);
			opennn::type border_for_12s = GetBorder12sValue(single_round_results, ordered_results);

			RecalculateModelTestStatsAndStore("new_trained", ("new_model"), single_round_results, amount_of_draws_in_data, buff_container, round_lottery_expect, border_for_top_value, border_for_3s, border_for_6s, border_for_12s);
		}

		SingleModelResult results;
		results.optMeth = optimizationMethod;
		results.lossMeth = lossMethod;
		for (const auto& el : buff_container)
		{
			results.efficiency_only_top += el.sum_average_top_1 / static_cast<float>(amount_of_draws_in_data);
			results.points_only_top += el.sum_avgPoints_top_1 / static_cast<float>(amount_of_draws_in_data);
			results.efficiency_of_1_high += el.sum_average_high_1 / static_cast<float>(amount_of_draws_in_data);
			results.efficiency_of_any_1 += el.sum_average_any_1 / static_cast<float>(amount_of_draws_in_data);
			results.efficiency_any_3 += el.sum_average_3 / static_cast<float>(amount_of_draws_in_data);
			results.efficiency_any_6 += el.sum_average_6 / static_cast<float>(amount_of_draws_in_data);
			results.efficiency_any_12 += el.sum_average_12 / static_cast<float>(amount_of_draws_in_data);
		}

		return results;
	}
	catch (const exception& e)
	{
		cout << e.what() << endl;

		SingleModelResult results;
		results.optMeth = optimizationMethod;
		results.lossMeth = lossMethod;
		results.efficiency_only_top = 0.0f;
		results.points_only_top = 0.0f;
		results.efficiency_of_1_high = 0.0f;
		results.efficiency_of_any_1 = 0.0f;
		results.efficiency_any_3 = 0.0f;
		results.efficiency_any_6 = 0.0f;
		results.efficiency_any_12 = 0.0f;

		return results;
	}
}
//----------------------------------------------------------
Tensor<type, 2> DataManager::TestWithData(std::shared_ptr<NeuralNetwork> neuralNetwork)
{
	Tensor<type, 2> inputs(m_testData.size(), neuralNetwork->get_inputs_number());
	Tensor<type, 2> outputs(m_testData.size(), neuralNetwork->get_outputs_number());

	Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
	Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

	inputs.setValues(m_testData);

	outputs = neuralNetwork->calculate_outputs(inputs.data(), inputs_dimensions);

	return outputs;
}
//----------------------------------------------------------
opennn::type DataManager::GetBorderTopValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
{
	if (orderedResults.empty())
		return 0;

	return orderedResults.at(0);
}
//----------------------------------------------------------
opennn::type DataManager::GetBorder3sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
{
	opennn::type response = 0;
	int counter = 0;
	for (const auto& high_val : orderedResults)
	{
		for (const auto& el : singleResults)
		{
			if (el == high_val)
			{
				counter++;
			}

			if (counter >= 3)
			{
				response = high_val;
				return response;
			}
		}
	}

	return response;
}
//----------------------------------------------------------
opennn::type DataManager::GetBorder6sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
{
	opennn::type border_for_6s = 0;
	int border_for_6s_counter = 0;
	for (const auto& high_val : orderedResults)
	{
		for (const auto& el : singleResults)
		{
			if (el == high_val)
			{
				border_for_6s_counter++;
			}

			if (border_for_6s_counter >= 6)
			{
				border_for_6s = high_val;
				break;
			}
		}

		if (border_for_6s_counter >= 6)
		{
			border_for_6s = high_val;
			break;
		}
	}

	return border_for_6s;
}
//----------------------------------------------------------
opennn::type DataManager::GetBorder12sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
{
	opennn::type response = 0;
	int counter = 0;
	for (const auto& high_val : orderedResults)
	{
		for (const auto& el : singleResults)
		{
			if (el == high_val)
			{
				counter++;
			}

			if (counter >= 12)
			{
				response = high_val;
				return response;
			}
		}
	}

	return response;
}
//----------------------------------------------------------
bool DataManager::IsFulfillExpectations(const SingleModelResult& results, const TrainingExpectations& expectations)
{
	if (results.efficiency_only_top < expectations.efficiency_only_top || results.points_only_top < expectations.points_only_top ||
		results.efficiency_of_1_high < expectations.efficiency_of_1_high || results.efficiency_of_any_1 < expectations.efficiency_of_any_1 ||
		results.efficiency_any_3 < expectations.efficiency_any_3 /*|| results.top3sPoints < expectations.points3sLevel*/ ||
		results.efficiency_any_6 < expectations.efficiency_any_6 /*|| results.top6sPoints < expectations.points6sLevel*/ ||
		results.efficiency_any_12 < expectations.efficiency_any_12 /*|| results.top12sPoints < expectations.points12sLevel*/)
	{
		return false;
	}

	return true;
}
//----------------------------------------------------------
void DataManager::SaveTrainingParameters(const MODEL_DESTINATION modelDestination, const TrainingParameters& trainingParameters, const std::string& saveFolder)
{
	const std::string file_name = saveFolder + "/" + GetModelDestinantionName(modelDestination) + "_" + "configuration.txt";
	std::fstream created_file;
	created_file.open(file_name, std::ios::out);

	if (!created_file)
	{
		std::cout << "[ERROR] " << file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << file_name << " created successfully!" << std::endl;

		created_file << "hiddenNeuronsNumber: " << trainingParameters.hiddenNeuronsNumber << "\n";
		created_file << "points_only_top: " << trainingParameters.expectations.efficiency_only_top << "\n";
		created_file << "efficiency_of_1_high: " << trainingParameters.expectations.points_only_top << "\n";
		created_file << "efficiency_of_any_1: " << trainingParameters.expectations.efficiency_of_1_high << "\n";
		created_file << "efficiency_any_3: " << trainingParameters.expectations.efficiency_of_any_1 << "\n";
		created_file << "efficiency_any_6: " << trainingParameters.expectations.efficiency_any_3 << "\n";
		created_file << "efficiency_any_12: " << trainingParameters.expectations.efficiency_any_6 << "\n";
	}
}
//----------------------------------------------------------
void DataManager::RecalculateModelTestStatsAndStore(const std::string& modelLocation, const std::string& modelName, const std::vector<float>& data, const int& numberOfRounds, std::set<TempSumOfModelTestStats>& buffContainer, const std::vector<int>& expectedResults, const opennn::type& border_for_top_value, const opennn::type& border_for_3s, const opennn::type& border_for_6s, const opennn::type& border_for_12s)
{
	ModelTestStats input_stats;

	for (int index = 0; index < expectedResults.size(); ++index)
	{
		if (data.at(index) == 1.0f)
		{
			input_stats.only_top_select++;

			if (expectedResults.at(index) > 0)
			{
				input_stats.only_top_hit++;

				input_stats.only_top_points = input_stats.only_top_points + expectedResults.at(index);
			}
		}

		if (data.at(index) >= border_for_top_value && data.at(index) >= m_limitValues.at(0))
		{
			input_stats.high_1_select++;

			if (expectedResults.at(index) > 0)
			{
				input_stats.high_1_hit++;

				input_stats.high_1_points = input_stats.high_1_points + expectedResults.at(index);
			}
		}

		if (data.at(index) >= border_for_top_value)
		{
			input_stats.any_1_select++;

			if (expectedResults.at(index) > 0)
			{
				input_stats.any_1_hit++;

				input_stats.any_1_points = input_stats.any_1_points + expectedResults.at(index);
			}
		}

		if (data.at(index) >= border_for_3s)
		{
			input_stats.select_3s++;

			if (expectedResults.at(index) > 0)
			{
				input_stats.hit_3s++;
			}
		}

		if (data.at(index) >= border_for_6s)
		{
			input_stats.select_6s++;

			if (expectedResults.at(index) > 0)
			{
				input_stats.hit_6s++;
			}
		}

		if (data.at(index) >= border_for_12s)
		{
			input_stats.select_12s++;

			if (expectedResults.at(index) > 0)
			{
				input_stats.hit_12s++;
			}
		}
	}

	float avg_only_top = input_stats.only_top_select > 0 ? input_stats.only_top_hit / static_cast<float>(input_stats.only_top_select) : 0.f;
	float avg_points_olnly_top = (input_stats.only_top_select * input_stats.only_top_hit * input_stats.only_top_points) > 0 ? (input_stats.only_top_points / static_cast<float>(input_stats.only_top_hit)) / static_cast<float>(input_stats.only_top_select) : 0.f;

	float avg_high_1 = input_stats.high_1_select > 0 ? input_stats.high_1_hit / static_cast<float>(input_stats.high_1_select) : 0.f;
	float avg_any_1 = input_stats.any_1_select > 0 ? input_stats.any_1_hit / static_cast<float>(input_stats.any_1_select) : 0.f;

	float avg_3s = input_stats.select_3s > 0 ? input_stats.hit_3s / static_cast<float>(input_stats.select_3s) : 0.f;

	float avg_6s = input_stats.select_6s > 0 ? input_stats.hit_6s / static_cast<float>(input_stats.select_6s) : 0.f;

	float avg_12s = input_stats.select_12s > 0 ? input_stats.hit_12s / static_cast<float>(input_stats.select_12s) : 0.f;

	TempSumOfModelTestStats obj({
						modelLocation,
						modelName,
						avg_only_top,
						avg_points_olnly_top,
						avg_high_1,
						avg_any_1,
						avg_3s,
						avg_6s,
						avg_12s,
		});

	if (auto it = buffContainer.find(obj); it != buffContainer.end())
	{
		auto buff = *it;
		buffContainer.erase(it);
		buff.sum_average_top_1 += obj.sum_average_top_1;
		buff.sum_avgPoints_top_1 += obj.sum_avgPoints_top_1;
		buff.sum_average_high_1 += obj.sum_average_high_1;
		buff.sum_average_any_1 += obj.sum_average_any_1;
		buff.sum_average_3 += obj.sum_average_3;
		buff.sum_average_6 += obj.sum_average_6;
		buff.sum_average_12 += obj.sum_average_12;

		buffContainer.insert(buff);
	}
	else
	{
		buffContainer.insert(obj);
	}
}
