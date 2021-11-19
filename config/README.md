# Config file
For each dataset a config file is defined. It is a json file that completely define the NAS process.
Each config file presents the following structure:
```json
{
	"name": "{dataset}_{model_name}",
	"n_gpu": {number of gpus},

	"arch": {
		"type": "{model_name}",
		"dataset": "{dataset}",
		"args": {
			{input args for model creation}
		}
	},
	"nas": {
		"do_nas": {true or false},
		"nas_config":{
			"schedule":{ # leave as-is
				"channels": "same",
				"filters": "same",
				"dilation": "same"
			}, 
			"target":{
				"dilation": {true or false}},
				"filters": {true or false},
				"channels": {true or false},} 
			},
			"mask_type":{
				"type": "{sum or mul}",
				"dilation": "binary", # leave as-is
				"filters": "binary", # leave as-is
				"channels": "binary" # leave as-is
			},
			"hysteresis":{ # leave as-is
				"do": false, 
				"eps": 0.0
			},
			"group_binarize": true, # leave as-is
			"regularizer": "{size or flops}",
			"rf_expansion_ratio": {integer, normally 1 or 2},
			"granularity": {integer, normally 1 or 2},
			"tau": 0.001, # leave as-is
			"strength":{
				"fixed": true, # leave as-is
				"value": {float}
			},
			"warmup_epochs": "max" # leave as-is
		}
	},
	"data_loader": {
		"type": "{data_loader_name}",
		"args":{
			"data_dir": "{dataset_path}",
			"batch_size": {integer},
			"shuffle": true,
			"validation_split": {fraction, e.g., 0.1},
			"num_workers": {integer}
		}
	},
	"optimizer": {
		"type": "{optimizer_name}",
		"args":{
			"lr": {float},
			"weight_decay": {float},
			"amsgrad": false
		}
	},
	"loss": "{loss_name}",
	"metrics": [
		"{metric_name}"
	],

    "trainer": {
		"type": "{trainer_name}",
		"epochs": {integer},
		"save_dir": "{output_path}",
		"save_period": {integer},
		"verbosity": 2,
		
		"monitor": "{min, max} {metric_name}",
		"early_stop": {integer},

		"tensorboard": false,
		"cross_validation": {
			"do": {true or false},
			"folds": {integer, depends on the specific datase}
		}
	}
}
```