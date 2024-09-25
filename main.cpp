
#include "Analizer.h"

int main()
{
	Analizer an("data/");

    an.SetTrainigParams(100, 0.75f, 3, 0.4f, 4);
	//an.TestSingleModel("7089", "together/models_all", "GD_MSE_2_best");
    an.StartTraining("7094", "1");
	//an.TestGivenModelsWithDatas({ "7093", "7094", "7095" }, "together/models_all");
	/*an.GetTestResultsForNewDraw("7096", "together/models_all",
		{
			"QN_SS_1_1_7092_100",
			"SG_MS_2_5_7092_100_best_4",
			"SG_CE_2_5_7092_100_best_4",
			"SG_SS_1_6_7092_100_best_5",
			"SG_CE_2_6_7092_100_best_5"
		});*/
	
}
