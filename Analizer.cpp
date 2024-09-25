#include "Analizer.h"

Analizer::Analizer(const std::string& rootFolder)
	:m_rootFolder(rootFolder),
	m_hiddenNeuronsNumber(100),
	m_selection_point(0.75f),
	m_number_of_repeates(3),
	m_acceptModelConditionParam(0.0f),
	m_topHitLevel(1)
{
}

void Analizer::SetTrainigParams(const Index& hiddenNeuronsNumber, const float& selectionPoint, const int& number_of_repeates, const float& acceptModelConditionParam, const int& topHitLevel)
{
	m_hiddenNeuronsNumber = hiddenNeuronsNumber;
	m_selection_point = selectionPoint;
	m_number_of_repeates = number_of_repeates;
	m_acceptModelConditionParam = acceptModelConditionParam;
	m_topHitLevel = topHitLevel;

	m_optimizationMethods = {
		TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT,
		TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT,
		TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD,
		/*TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM,*/
		TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT,
		TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION
	};
	m_lossMethods = {
		TrainingStrategy::LossMethod::SUM_SQUARED_ERROR,
		TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR,
		TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR,
		/*TrainingStrategy::LossMethod::MINKOWSKI_ERROR,*/
		TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR,
		TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR
	};
}

void Analizer::StartTraining(const std::string& locationFolder, const std::string& tryPostFix)
{
	std::string models_folder_name = m_rootFolder + locationFolder + "/models_" + tryPostFix;
	CreateModelsFolder(models_folder_name);

	std::string data_file_name = m_rootFolder + locationFolder + "/input_data.csv";

	const std::string file_sufix = tryPostFix + "_" + locationFolder + "_" + std::to_string(m_hiddenNeuronsNumber);

	m_dataSet.set(data_file_name, ';', true);

	m_inputVariablesNumber = m_dataSet.get_input_variables_number();
	m_targetVariablesNumber = m_dataSet.get_target_variables_number();

	LoadTestInputData(locationFolder);
	LoadTestExpectedResults(locationFolder);

	std::map<std::string, std::vector<SingleResult>> tests_summarize;

	for (const auto& optimization_method : m_optimizationMethods)
	{
		for (const auto& loss_method : m_lossMethods)
		{
			int repeate_counter = 0;
			const std::string combination_name = GetTestFileName(optimization_method, loss_method);
			while (repeate_counter < m_number_of_repeates)
			{
				auto neural_network = GetTrainedNeuralNetwork(NeuralNetwork::ProjectType::Classification, m_hiddenNeuronsNumber, optimization_method, loss_method);

				if (neural_network)
				{
					auto resp = CollectResult(neural_network, optimization_method, loss_method);

					if (!resp.empty())
					{
						repeate_counter++;
						std::string try_name = combination_name + "_" + std::to_string(repeate_counter);

						tests_summarize[try_name] = resp;

						std::string file_name = models_folder_name + "/ " + try_name + "_" + file_sufix + "_best_" + std::to_string(m_topHitLevel) + ".xml";

						neural_network->save(file_name);
					}
				}
			}
		}
	}

	const std::string folder = m_rootFolder + locationFolder + "/";
	const std::string file_name = folder + "training_test_results.txt";
	std::fstream created_file;
	created_file.open(file_name, std::ios::out);

	if (!created_file)
	{
		std::cout << "[ERROR] " << file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << file_name << " created successfully!" << std::endl;

		for (const auto& [key, value] : tests_summarize)
		{
			for (const auto& el : value)
			{
				created_file << key << ";";
			}
			created_file << "\n";
			for (const auto& el : value)
			{
				created_file << el.selected << ";";
			}
			created_file << "\n";
			for (const auto& el : value)
			{
				created_file << el.hits << ";";
			}
			created_file << "\n";
			for (const auto& el : value)
			{
				created_file << el.points << ";";
			}
			created_file << "\n";
		}
	}
}

void Analizer::StartTrainingWithVecOfParams(const std::string& locationFolder, const std::string& tryPostFix, const std::vector<TrainingParams>& vecOfParams)
{
	std::string models_folder_name = m_rootFolder + locationFolder + "/models_" + tryPostFix;
	CreateModelsFolder(models_folder_name);

	std::string data_file_name = m_rootFolder + locationFolder + "/input_data.csv";

	m_dataSet.set(data_file_name, ';', true);

	m_inputVariablesNumber = m_dataSet.get_input_variables_number();
	m_targetVariablesNumber = m_dataSet.get_target_variables_number();

	LoadTestInputData(locationFolder);
	LoadTestExpectedResults(locationFolder);

	std::map<std::string, std::vector<SingleResult>> tests_summarize;

	for (const auto& params : vecOfParams)
	{
		int repeate_counter = 0;
		const std::string combination_name = GetTestFileName(params.optMeth, params.lossMeth);
		while (repeate_counter < m_number_of_repeates)
		{
			auto neural_network = GetTrainedNeuralNetwork(NeuralNetwork::ProjectType::Classification, m_hiddenNeuronsNumber, params.optMeth, params.lossMeth);

			if (neural_network)
			{
				auto resp = CollectResult(neural_network, params.optMeth, params.lossMeth);

				if (!resp.empty())
				{
					repeate_counter++;
					std::string try_name = combination_name + "_" + std::to_string(repeate_counter);

					tests_summarize[try_name] = resp;

					std::string file_name = models_folder_name + "/ " + try_name + ".xml";

					neural_network->save(file_name);
				}
			}
		}
	}

	const std::string folder = m_rootFolder + locationFolder + "/";
	const std::string file_name = folder + "training_test_results.txt";
	std::fstream created_file;
	created_file.open(file_name, std::ios::out);

	if (!created_file)
	{
		std::cout << "[ERROR] " << file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << file_name << " created successfully!" << std::endl;

		for (const auto& [key, value] : tests_summarize)
		{
			for (const auto& el : value)
			{
				created_file << key << ";";
			}
			created_file << "\n";
			for (const auto& el : value)
			{
				created_file << el.selected << ";";
			}
			created_file << "\n";
			for (const auto& el : value)
			{
				created_file << el.hits << ";";
			}
			created_file << "\n";
			for (const auto& el : value)
			{
				created_file << el.points << ";";
			}
			created_file << "\n";
		}
	}
}

void Analizer::TestSingleModel(const std::string& testDataFolder, const std::string& modelsFolder, const std::string& modelName)
{
	LoadTestInputData(testDataFolder);
	LoadTestExpectedResults(testDataFolder);

	try
	{
		const std::string path_and_name = modelsFolder + "/" + modelName;
		auto neural_network = LoadNeuralNetwork(path_and_name);

		if (neural_network)
		{
			auto results = TestWithData(neural_network);

			std::vector<float> single_results;

			for (int i = 0; i < results.dimension(0); ++i)
			{
				for (int j = 0; j < results.dimension(1); ++j)
				{
					single_results.push_back(results(i, j));
				}
			}

			std::set<float> buff_set_of_results(single_results.begin(), single_results.end());
			std::vector<float> ordered_results(buff_set_of_results.begin(), buff_set_of_results.end());
			std::reverse(std::begin(ordered_results), std::end(ordered_results));

			int selected_numbers = 0;
			int hited_numbers = 0;
			int sum_of_points = 0;

			int top_numbers = 0;
			int hited_from_top = 0;
			int points_from_top = 0;

			opennn::type border_for_top_value = GetBorderTopValue(single_results, ordered_results);
			opennn::type border_for_3s = GetBorder3sValue(single_results, ordered_results);
			opennn::type border_for_6s = GetBorder6sValue(single_results, ordered_results);
			opennn::type border_for_12s = GetBorder12sValue(single_results, ordered_results);

			for (int index = 0; index < m_expectedResults.size(); ++index)
			{
				if (single_results.at(index) > m_selection_point)
				{
					selected_numbers++;

					if (m_expectedResults.at(index) > 0)
					{
						hited_numbers++;

						sum_of_points = sum_of_points + m_expectedResults.at(index);
					}
				}

				if (single_results.at(index) >= border_for_6s)
				{
					top_numbers++;

					if (m_expectedResults.at(index) > 0)
					{
						hited_from_top++;

						points_from_top = points_from_top + m_expectedResults.at(index);
					}
				}
			}

			SingleTestResult test_result_1;
			test_result_1.testName = modelName;
			test_result_1.selected = selected_numbers;
			test_result_1.hits = hited_numbers;
			test_result_1.points = sum_of_points;
			test_result_1.results = single_results;
			test_result_1.expected = m_expectedResults;

			float avg_hits_1 = selected_numbers > 0 ? hited_numbers / static_cast<float>(selected_numbers) : 0.f;
			float points_per_hits_1 = selected_numbers > 0 ? (hited_numbers * sum_of_points) / static_cast<float>(selected_numbers) : 0.f;
			float avg_plus_ph_1 = avg_hits_1 + (points_per_hits_1 / 10.f);

			SingleTestResult test_result_2;
			test_result_2.testName = modelName;
			test_result_2.selected = top_numbers;
			test_result_2.hits = hited_from_top;
			test_result_2.points = points_from_top;
			test_result_2.results = single_results;
			test_result_2.expected = m_expectedResults;

			float avg_hits_2 = top_numbers > 0 ? hited_from_top / static_cast<float>(top_numbers) : 0.f;
			float points_per_hits_2 = top_numbers > 0 ? (hited_from_top * points_from_top) / static_cast<float>(top_numbers) : 0.f;
			float avg_plus_ph_2 = avg_hits_2 + (points_per_hits_2 / 10.f);

			float value_grade = (avg_plus_ph_1 + avg_plus_ph_2) / 2.f;

			std::cout << "RESULTS of data from " << testDataFolder << " on model " << path_and_name << " :" << std::endl;
			std::cout << "\t" << " [pointed] \t [first 6s] " << std::endl;
			std::cout << "[SELECTED]:\t" << selected_numbers << "\t" << top_numbers << std::endl;
			std::cout << "[HITS]:\t\t" << hited_numbers << "\t" << hited_from_top << std::endl;
			std::cout << "[POINTS]:\t" << sum_of_points << "\t" << points_from_top << std::endl;
			std::cout << std::endl;
			std::cout << "[AVG HITS]:\t" << avg_hits_1 << "\t" << avg_hits_2 << std::endl;
			std::cout << "[P/HITS]:\t" << points_per_hits_1 << "\t" << points_per_hits_2 << std::endl;
			std::cout << "[AVG+P/h]:\t" << avg_plus_ph_1 << "\t" << avg_plus_ph_2 << std::endl;
			std::cout << "[VALUE GRADE]:  \t" << value_grade << std::endl;
			std::cout << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}
}

void Analizer::TestDataWithModels(const std::string& testDataFolder, const std::string& modelsFolder, const std::vector<std::string>& modelsNames)
{
	LoadTestInputData(testDataFolder);
	LoadTestExpectedResults(testDataFolder);

	std::map<std::string, std::map<std::string, std::array<std::vector<SingleTestResult>, 3U>>> tests_summarize_2 =
	{
		{"GD", { { "SS", {} }, { "MS", {} }, { "NS", {} }, /*{ "ME", {} },*/ { "WQ", {} }, { "CE", {} } }},
		{"CG", { { "SS", {} }, { "MS", {} }, { "NS", {} }, /*{ "ME", {} },*/ { "WQ", {} }, { "CE", {} } }},
		{"QN", { { "SS", {} }, { "MS", {} }, { "NS", {} }, /*{ "ME", {} },*/ { "WQ", {} }, { "CE", {} } }},
		/*{"LM", { { "SS", {} }, { "MS", {} }, { "NS", {} }, { "ME", {} }, { "WQ", {} }, { "CE", {} } }},*/
		{"SG", { { "SS", {} }, { "MS", {} }, { "NS", {} }, /*{ "ME", {} },*/ { "WQ", {} }, { "CE", {} } }},
		{"AM", { { "SS", {} }, { "MS", {} }, { "NS", {} }, /*{ "ME", {} },*/ { "WQ", {} }, { "CE", {} } }},
	};

	for (const auto& name : modelsNames)
	{
		try
		{
			const std::string path_and_name = modelsFolder + "/" + name;
			auto neural_network = LoadNeuralNetwork(path_and_name);

			if (neural_network)
			{
				auto results = TestWithData(neural_network);

				std::vector<float> single_results;

				for (int i = 0; i < results.dimension(0); ++i)
				{
					for (int j = 0; j < results.dimension(1); ++j)
					{
						single_results.push_back(results(i, j));
					}
				}

				std::set<float> buff_set_of_results(single_results.begin(), single_results.end());
				std::vector<float> ordered_results(buff_set_of_results.begin(), buff_set_of_results.end());
				std::reverse(std::begin(ordered_results), std::end(ordered_results));

				int selected_numbers = 0;
				int hited_numbers = 0;
				int sum_of_points = 0;

				int top_numbers = 0;
				int hited_from_top = 0;
				int points_from_top = 0;

				opennn::type border_for_top_value = GetBorderTopValue(single_results, ordered_results);
				opennn::type border_for_3s = GetBorder3sValue(single_results, ordered_results);
				opennn::type border_for_6s = GetBorder6sValue(single_results, ordered_results);
				opennn::type border_for_12s = GetBorder12sValue(single_results, ordered_results);

				for (int index = 0; index < m_expectedResults.size(); ++index)
				{
					if (single_results.at(index) > m_selection_point)
					{
						selected_numbers++;

						if (m_expectedResults.at(index) > 0)
						{
							hited_numbers++;

							sum_of_points = sum_of_points + m_expectedResults.at(index);
						}
					}

					if (single_results.at(index) >= border_for_6s)
					{
						top_numbers++;

						if (m_expectedResults.at(index) > 0)
						{
							hited_from_top++;

							points_from_top = points_from_top + m_expectedResults.at(index);
						}
					}
				}

				std::stringstream ss(name);
				std::vector<std::string> buff;

				while (ss.good())
				{
					std::string substr;
					std::getline(ss, substr, '_');
					buff.push_back(substr);
				}

				SingleTestResult test_result_1;
				test_result_1.testName = name;
				test_result_1.selected = selected_numbers;
				test_result_1.hits = hited_numbers;
				test_result_1.points = sum_of_points;
				test_result_1.results = single_results;
				test_result_1.expected = m_expectedResults;

				SingleTestResult test_result_2;
				test_result_2.testName = name;
				test_result_2.selected = top_numbers;
				test_result_2.hits = hited_from_top;
				test_result_2.points = points_from_top;
				test_result_2.results = single_results;
				test_result_2.expected = m_expectedResults;

				tests_summarize_2.at(buff.at(0)).at(buff.at(1)).at(std::stoi(buff.at(2)) - 1).push_back(test_result_1);
				tests_summarize_2.at(buff.at(0)).at(buff.at(1)).at(std::stoi(buff.at(2)) - 1).push_back(test_result_2);
			}
		}
		catch (const std::exception& e)
		{
			cout << "[" << name << "] " << e.what() << endl;
		}
	}

	const std::string test_folder_name = m_rootFolder + GetTestFolderName(testDataFolder, modelsFolder);

	CreateTestFolder(test_folder_name);

	const std::string list_of_models_file_name = test_folder_name + "/list_of_models.txt";
	std::fstream created_list_file;
	created_list_file.open(list_of_models_file_name, std::ios::out);

	if (!created_list_file)
	{
		std::cout << "[ERROR] " << list_of_models_file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << list_of_models_file_name << " created successfully!" << std::endl;

		created_list_file << "tested models:\n";

		for (const auto& el : modelsNames)
		{
			created_list_file << el << "\n";
		}
	}

	const std::string file_name = test_folder_name + "/models_tests_results.txt";
	std::fstream created_file;
	created_file.open(file_name, std::ios::out);

	if (!created_file)
	{
		std::cout << "[ERROR] " << file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << file_name << " created successfully!" << std::endl;

		bool columns_printed = false;
		for (const auto& [key_1, val_1] : tests_summarize_2)
		{
			if (!columns_printed)
			{
				created_file << ";";
				for (const auto& [key_2, val_2] : val_1)
				{
					for (int i = 1; i <= val_2.size(); ++i)
					{
						std::string n = key_2 + "_" + std::to_string(i);

						created_file << n << ";" << n << ";";
					}
				}
				created_file << "\n";
				columns_printed = true;
			}

			created_file << key_1 << ";";
			for (const auto& [key_2, val_2] : val_1)
			{
				for (const auto& el : val_2)
				{
					for (const auto& e : el)
					{
						created_file << e.selected << ";";
					}
				}
			}
			created_file << "\n";
			created_file << key_1 << ";";
			for (const auto& [key_2, val_2] : val_1)
			{
				for (const auto& el : val_2)
				{
					for (const auto& e : el)
					{
						created_file << e.hits << ";";
					}
				}
			}
			created_file << "\n";
			created_file << key_1 << ";";
			for (const auto& [key_2, val_2] : val_1)
			{
				for (const auto& el : val_2)
				{
					for (const auto& e : el)
					{
						created_file << e.points << ";";
					}
				}
			}
			created_file << "\n";
		}
	}

	const std::string check_tables_file_name = test_folder_name + "/" + GetExcelFileName(testDataFolder, modelsFolder);
	std::fstream check_table_file;
	check_table_file.open(check_tables_file_name, std::ios::out);

	if (!check_table_file)
	{
		std::cout << "[ERROR] " << check_tables_file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << check_tables_file_name << " created successfully!" << std::endl;
	}
}

void Analizer::GetTestResultsForNewDraw(const std::string& testDataFolder, const std::string& modelsFolder, const std::vector<std::string>& modelsNames)
{
	LoadTestInputData(testDataFolder);
	LoadTestExpectedResults(testDataFolder);

	if (m_expectedResults.empty())
	{
		m_expectedResults = std::vector<int>(49, 0);
	}

	std::vector<SingleTestResult> tests_summarize;

	for (const auto& name : modelsNames)
	{
		try
		{
			const std::string path_and_name = modelsFolder + "/" + name;
			auto neural_network = LoadNeuralNetwork(path_and_name);

			if (neural_network)
			{
				auto results = TestWithData(neural_network);

				std::vector<float> single_results;

				for (int i = 0; i < results.dimension(0); ++i)
				{
					for (int j = 0; j < results.dimension(1); ++j)
					{
						single_results.push_back(results(i, j));
					}
				}

				SingleTestResult test_result;
				test_result.testName = name;
				test_result.selected = 0;
				test_result.hits = 0;
				test_result.points = 0;
				test_result.results = single_results;
				test_result.expected = m_expectedResults;

				tests_summarize.push_back(test_result);
			}
		}
		catch (const std::exception& e)
		{
			cout << e.what() << endl;
		}
	}

	const std::string test_folder_name = m_rootFolder + GetTestFolderName(testDataFolder, modelsFolder);

	CreateTestFolder(test_folder_name);

	const std::string list_of_models_file_name = test_folder_name + "/list_of_models_used_for_draw.txt";
	std::fstream created_list_file;
	created_list_file.open(list_of_models_file_name, std::ios::out);

	if (!created_list_file)
	{
		std::cout << "[ERROR] " << list_of_models_file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << list_of_models_file_name << " created successfully!" << std::endl;

		created_list_file << "tested models:\n";

		for (const auto& el : modelsNames)
		{
			created_list_file << el << "\n";
		}
	}

	const int number_of_rows = static_cast<int>(m_expectedResults.size());

	const std::string file_name = test_folder_name + "/models_tests_results_for_draw.txt";
	std::fstream created_file;
	created_file.open(file_name, std::ios::out);

	if (!created_file)
	{
		std::cout << "[ERROR] " << file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << file_name << " created successfully!" << std::endl;

		for (const auto& el : tests_summarize)
		{
			created_file << (el.testName + "_values") << ";" << (el.testName + "_expect") << ";;;;";
		}

		created_file << "\n";

		for (int i = 0; i < number_of_rows; ++i)
		{
			for (const auto& el : tests_summarize)
			{
				created_file << el.results.at(i) << ";" << el.expected.at(i) << ";;;;";
			}

			created_file << "\n";
		}

		created_file << "\n";
	}
}

void Analizer::TestGivenModelsWithDatas(const std::vector<std::string>& containerOfDatas, const std::string& modelsFolder)
{
	std::vector<std::string> models_names;
	for (const auto& entry : fs::directory_iterator(m_rootFolder + modelsFolder))
	{
		// Converting the path to const char * in the
		// subsequent lines
		std::filesystem::path outfilename = entry.path();
		std::string outfilename_str = outfilename.string();
		std::size_t found = outfilename_str.find("\\");
		if (found != std::string::npos)
		{
			models_names.push_back(std::string(outfilename_str.begin() + (found + 1), outfilename_str.end() - 4));
		}
	}

	std::multiset<SumOfTests> results;
	std::set<SumOfTests_1> buff_result;

	for (const auto& dataFolder : containerOfDatas)
	{
		LoadTestInputData(dataFolder);
		LoadTestExpectedResults(dataFolder);

		for (const auto& modelName : models_names)
		{
			try
			{
				const std::string path_and_name = modelsFolder + "/" + modelName;
				auto neural_network = LoadNeuralNetwork(path_and_name);

				if (neural_network)
				{
					auto results = TestWithData(neural_network);

					std::vector<float> single_results;

					for (int i = 0; i < results.dimension(0); ++i)
					{
						for (int j = 0; j < results.dimension(1); ++j)
						{
							single_results.push_back(results(i, j));
						}
					}

					std::set<float> buff_set_of_results(single_results.begin(), single_results.end());
					std::vector<float> ordered_results(buff_set_of_results.begin(), buff_set_of_results.end());
					std::reverse(std::begin(ordered_results), std::end(ordered_results));

					int top_select = 0;
					int top_hit = 0;
					int top_points = 0;

					int select_3s = 0;
					int hit_3s = 0;
					int points_3s = 0;

					int select_6s = 0;
					int hit_6s = 0;
					int points_6s = 0;

					int select_12s = 0;
					int hit_12s = 0;
					int points_12s = 0;

					/*int selected_numbers = 0;
					int hited_numbers = 0;
					int sum_of_points = 0;

					int top_numbers = 0;
					int hited_from_top = 0;
					int points_from_top = 0;*/

					opennn::type border_for_top_value = GetBorderTopValue(single_results, ordered_results);
					opennn::type border_for_3s = GetBorder3sValue(single_results, ordered_results);
					opennn::type border_for_6s = GetBorder6sValue(single_results, ordered_results);
					opennn::type border_for_12s = GetBorder12sValue(single_results, ordered_results);

					for (int index = 0; index < m_expectedResults.size(); ++index)
					{
						if (single_results.at(index) >= border_for_top_value)
						{
							top_select++;

							if (m_expectedResults.at(index) > 0)
							{
								top_hit++;

								top_points = top_points + m_expectedResults.at(index);
							}
						}

						if (single_results.at(index) >= border_for_3s)
						{
							select_3s++;

							if (m_expectedResults.at(index) > 0)
							{
								hit_3s++;

								points_3s = points_3s + m_expectedResults.at(index);
							}
						}

						if (single_results.at(index) >= border_for_6s)
						{
							select_6s++;

							if (m_expectedResults.at(index) > 0)
							{
								hit_6s++;

								points_6s = points_6s + m_expectedResults.at(index);
							}
						}

						if (single_results.at(index) >= border_for_12s)
						{
							select_12s++;

							if (m_expectedResults.at(index) > 0)
							{
								hit_12s++;

								points_12s = points_12s + m_expectedResults.at(index);
							}
						}
					}

					float avg_top = top_select > 0 ? top_hit / static_cast<float>(top_select) : 0.f;
					float avg_points_top = (top_select * top_hit * top_points) > 0 ? (top_points / static_cast<float>(top_hit)) / static_cast<float>(top_select) : 0.f;

					float avg_3s = select_3s > 0 ? hit_3s / static_cast<float>(select_3s) : 0.f;
					float avg_points_3s = (select_3s * hit_3s * points_3s) > 0 ? (points_3s / static_cast<float>(hit_3s)) / static_cast<float>(select_3s) : 0.f;

					float avg_6s = select_6s > 0 ? hit_6s / static_cast<float>(select_6s) : 0.f;
					float avg_points_6s = (select_6s * hit_6s * points_6s) > 0 ? (points_6s / static_cast<float>(hit_6s)) / static_cast<float>(select_6s) : 0.f;

					float avg_12s = select_12s > 0 ? hit_12s / static_cast<float>(select_12s) : 0.f;
					float avg_points_12s = (select_12s * hit_12s * points_12s) > 0 ? (points_12s / static_cast<float>(hit_12s)) / static_cast<float>(select_12s) : 0.f;

					SumOfTests_1 obj({
						modelName,
						avg_top,
						avg_points_top,
						avg_3s,
						avg_points_3s,
						avg_6s,
						avg_points_6s,
						avg_12s,
						avg_points_12s
						});

					if (auto it = buff_result.find(obj); it != buff_result.end())
					{
						auto buff = *it;
						buff_result.erase(it);
						buff.sum_average_1 += obj.sum_average_1;
						buff.sum_average_3 += obj.sum_average_3;
						buff.sum_average_6 += obj.sum_average_6;
						buff.sum_average_12 += obj.sum_average_12;
						buff.sum_avgPoints_1 += obj.sum_avgPoints_1;
						buff.sum_avgPoints_3 += obj.sum_avgPoints_3;
						buff.sum_avgPoints_6 += obj.sum_avgPoints_6;
						buff.sum_avgPoints_12 += obj.sum_avgPoints_12;

						buff_result.insert(buff);
					}
					else
					{
						buff_result.insert(obj);
					}
				}
			}
			catch (const std::exception& e)
			{
				cout << "[" << dataFolder << "][" << modelName << "] " << e.what() << endl;
			}
		}
	}

	const int data_counter = static_cast<int>(containerOfDatas.size());

	for (const auto& el : buff_result)
	{
		SumOfTests new_el({
			el.name,
			(el.sum_average_1 / static_cast<float>(data_counter)),
			(el.sum_avgPoints_1 / static_cast<float>(data_counter)),
			(el.sum_average_3 / static_cast<float>(data_counter)),
			(el.sum_avgPoints_3 / static_cast<float>(data_counter)),
			(el.sum_average_6 / static_cast<float>(data_counter)),
			(el.sum_avgPoints_6 / static_cast<float>(data_counter)),
			(el.sum_average_12 / static_cast<float>(data_counter)),
			(el.sum_avgPoints_12 / static_cast<float>(data_counter)),
			});

		results.insert(new_el);
	}

	const std::string result_file_name = m_rootFolder + "MODELS_RESULT.txt";
	std::fstream created_file;
	created_file.open(result_file_name, std::ios::out);

	if (!created_file)
	{
		std::cout << "[ERROR] " << result_file_name << " not created!" << std::endl;
	}
	else
	{
		std::cout << result_file_name << " created successfully!" << std::endl;

		created_file << "NAME;AVERAGE_1;AVG_POINTS_1;AVERAGE_3;AVG_POINTS_3;AVERAGE_6;AVG_POINTS_6;AVERAGE_12;AVG_POINTS_12\n";

		for (const auto& el : results)
		{
			created_file << el.name << ";" << el.average_1 << ";" << el.avgPoints_1 << ";" << el.average_3 << ";" << el.avgPoints_3 << ";" << el.average_6 << ";" << el.avgPoints_6 << ";" << el.average_12 << ";" << el.avgPoints_12 << "\n";
		}
	}

	int a = 0;
}

std::shared_ptr<NeuralNetwork> Analizer::LoadNeuralNetwork(const std::string& nnName)
{
	const std::string file_name = m_rootFolder + nnName + ".xml";

	return std::make_shared<NeuralNetwork>(file_name);
}

std::shared_ptr<NeuralNetwork> Analizer::GetTrainedNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Index& hiddenNeuronsNumber, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod)
{
	try
	{
		srand(static_cast<unsigned>(time(nullptr)));

		std::shared_ptr<NeuralNetwork> neural_network(new NeuralNetwork(projectType, { m_inputVariablesNumber, hiddenNeuronsNumber, m_targetVariablesNumber }));

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

Tensor<type, 2> Analizer::TestWithData(std::shared_ptr<NeuralNetwork> neuralNetwork)
{
	Tensor<type, 2> inputs(49, neuralNetwork->get_inputs_number());
	Tensor<type, 2> outputs(49, neuralNetwork->get_outputs_number());

	Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
	Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

	inputs.setValues(m_testData);

	outputs = neuralNetwork->calculate_outputs(inputs.data(), inputs_dimensions);

	return outputs;
}

std::vector<SingleResult> Analizer::CollectResult(std::shared_ptr<NeuralNetwork> neuralNetwork, const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod)
{
	try
	{
		auto results = TestWithData(neuralNetwork);

		std::vector<float> single_results;

		for (int i = 0; i < results.dimension(0); ++i)
		{
			for (int j = 0; j < results.dimension(1); ++j)
			{
				single_results.push_back(results(i, j));
			}
		}

		std::set<float> buff_set_of_results(single_results.begin(), single_results.end());
		std::vector<float> ordered_results(buff_set_of_results.begin(), buff_set_of_results.end());
		std::reverse(std::begin(ordered_results), std::end(ordered_results));

		int selected_numbers = 0;
		int hited_numbers = 0;
		int sum_of_points = 0;

		int top_numbers = 0;
		int hited_from_top = 0;
		int points_from_top = 0;

		int top_select = 0;
		int top_hit = 0;
		int top_points = 0;

		opennn::type border_for_top_value = GetBorderTopValue(single_results, ordered_results);
		opennn::type border_for_3s = GetBorder3sValue(single_results, ordered_results);
		opennn::type border_for_6s = GetBorder6sValue(single_results, ordered_results);
		opennn::type border_for_12s = GetBorder12sValue(single_results, ordered_results);

		for (int index = 0; index < m_expectedResults.size(); ++index)
		{
			if (single_results.at(index) >= border_for_top_value)
			{
				top_select++;

				if (m_expectedResults.at(index) > 0)
				{
					top_hit++;

					top_points = top_points + m_expectedResults.at(index);
				}
			}
			if (single_results.at(index) > m_selection_point)
			{
				selected_numbers++;

				if (m_expectedResults.at(index) > 0)
				{
					hited_numbers++;

					sum_of_points = sum_of_points + m_expectedResults.at(index);
				}
			}

			if (single_results.at(index) >= border_for_6s)
			{
				top_numbers++;

				if (m_expectedResults.at(index) > 0)
				{
					hited_from_top++;

					points_from_top = points_from_top + m_expectedResults.at(index);
				}
			}
		}

		/*if (top_hit == 0)
		{
			return {};
		}*/

		if (!AcceptNNModelCondition(selected_numbers, hited_numbers, sum_of_points, top_numbers, hited_from_top, points_from_top))
		{
			return {};
		}
		//std::cout << std::endl;
		//cout << "\n[!!!]Results:\n" << results << endl;
		SingleResult from_selection_point;
		from_selection_point.optMeth = optimizationMethod;
		from_selection_point.lossMeth = lossMethod;
		from_selection_point.selected = selected_numbers;
		from_selection_point.hits = hited_numbers;
		from_selection_point.points = sum_of_points;

		SingleResult from_tops;
		from_tops.optMeth = optimizationMethod;
		from_tops.lossMeth = lossMethod;
		from_tops.selected = top_numbers;
		from_tops.hits = hited_from_top;
		from_tops.points = points_from_top;

		return { from_selection_point, from_tops };
	}
	catch (const exception& e)
	{
		cout << e.what() << endl;

		return {};
	}
}

void Analizer::LoadTestInputData(const std::string& givenFolderName)
{
	m_tempTestDataVoI.clear();
	m_tempTestDataVoV.clear();

	const std::string file_name = m_rootFolder + givenFolderName + "/test_input_data.txt";
	std::ifstream myfile(file_name);
	std::string mystring;

	int buff_index = 0;
	if (myfile.is_open())
	{
		while (myfile.good())
		{
			myfile >> mystring;

			m_tempTestDataVoV.push_back(std::vector<type>());
			std::stringstream ss(mystring);

			while (ss.good())
			{
				std::string substr;
				std::getline(ss, substr, ';');
				m_tempTestDataVoV.at(buff_index).push_back(type(std::stof(substr)));
			}
			//std::initializer_list<type> in(v.data(), v.data() + v.size());
            m_tempTestDataVoI.push_back(std::initializer_list<type>(m_tempTestDataVoV.at(buff_index).data(), m_tempTestDataVoV.at(buff_index).data() + m_tempTestDataVoV.at(buff_index).size()));
			++buff_index;
		}
	}
	else
	{
		std::cout << "[ERROR] Cannot open file " + file_name << std::endl;
	}
    m_testData = std::initializer_list<std::initializer_list<type>>(m_tempTestDataVoI.data(), m_tempTestDataVoI.data() + m_tempTestDataVoI.size());
}

void Analizer::LoadTestExpectedResults(const std::string& givenFolderName)
{
	m_expectedResults.clear();

	const std::string folder_name = givenFolderName;
	const std::string file_name = m_rootFolder + folder_name + "/expected_results.txt";
	std::ifstream myfile(file_name);
	std::string mystring;

	if (myfile.is_open())
	{
		while (myfile.good())
		{
			myfile >> mystring;

			m_expectedResults.push_back(std::stoi(mystring));
		}
	}
	else
	{
		std::cout << "[ERROR] Cannot open file " + file_name << std::endl;
	}
}

std::string Analizer::GetTestFileName(const TrainingStrategy::OptimizationMethod& optimizationMethod, const TrainingStrategy::LossMethod& lossMethod)
{
	std::string response = "";

	switch (optimizationMethod)
	{
	case TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT:
	{
		response = "GD_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT:
	{
		response = "CG_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD:
	{
		response = "QN_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM:
	{
		response = "LM_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT:
	{
		response = "SG_";
		break;
	}
	case TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION:
	{
		response = "AM_";
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

bool Analizer::AcceptNNModelCondition(const int& selectedNumbers, const int& hitedNumbers, const int& sumOfPoints, const int& topNumbers, const int& hitedFromTop, const int& pointsFromTop)
{
	if (hitedFromTop < m_topHitLevel)
	{
		//return false;
	}

	auto param = hitedFromTop / static_cast<float>(topNumbers);

	if (param < m_acceptModelConditionParam)
	{
		std::cout << "NOT ACCEPT: " << param << std::endl;
		return false;
	}

	std::cout << "[!!!] ACCEPTED: " << param << std::endl;

	return true;
}

std::string Analizer::GetTestFolderName(const std::string& testDataFolder, const std::string& modelsFolder)
{
	size_t pos = modelsFolder.find('/');
	std::string sub_str = modelsFolder;

	if (pos != std::string::npos)
	{
		std::replace(sub_str.begin(), sub_str.end(), '/', '_');
	}
	else
	{
		std::cout << "Character not found in the string." << std::endl;

		return "";
	}

	return "data_from_" + testDataFolder + "_models_from_" + sub_str;
}

std::string Analizer::GetExcelFileName(const std::string& testDataFolder, const std::string& modelsFolder)
{
	size_t pos = modelsFolder.find('/');
	std::string sub_str = modelsFolder;

	if (pos != std::string::npos)
	{
		std::replace(sub_str.begin(), sub_str.end(), '/', '_');
	}
	else
	{
		std::cout << "Character not found in the string." << std::endl;

		return "";
	}

	return "check_tables_" + testDataFolder + "_" + sub_str + ".txt";
}

void Analizer::CreateTestFolder(const std::string& testFolder)
{
	fs::create_directory(testFolder);

	if (fs::exists(testFolder))
	{
		std::cout << "Directory " << testFolder << " created successfully." << std::endl;
	}
	else
	{
		std::cout << "Failed to create directory " << testFolder << std::endl;
	}
}

void Analizer::CreateModelsFolder(const std::string& folderName)
{
	fs::create_directory(folderName);

	if (fs::exists(folderName))
	{
		std::cout << "Directory " << folderName << " created successfully." << std::endl;
	}
	else
	{
		std::cout << "Failed to create directory " << folderName << std::endl;
	}
}

opennn::type Analizer::GetBorderTopValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
{
	if (orderedResults.empty())
		return 0;

	return orderedResults.at(0);
}

opennn::type Analizer::GetBorder3sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
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

opennn::type Analizer::GetBorder6sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
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

opennn::type Analizer::GetBorder12sValue(const std::vector<float>& singleResults, const std::vector<float>& orderedResults)
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
