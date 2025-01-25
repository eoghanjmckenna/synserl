# synserl
Synthetic SERL Observatory data

`\notebooks\one_household_model.py` is the script I've developed which is a single self-contained file and which can be run alone and which randomly selects a puprn (unique household id) file from a specified folder that contains pickle pandas dataframes consisting of:

- `Read_date_effective_local` a datetime, half-hourly resolution
- `Clean_elec_imp_Wh` electricity imports,
- `Clean_gas_Wh` gas consumption
- `temp_C` external temperature

The script trains a tranformer model, and generates a synthetic version of the original. 

The model hyperparameters can be configured by changing parameters of the `config` dictionary on line 332 of the script.

The script should be run from the terminal with current working directory in the `notebooks` folder.

The script expects a folder `synserl\experiments\manual_search\` where it will automatically log metrics, plots, and trained model parameters in separate training run folders. A summary file of all the training runs and their performance is maintained in the `synserl\experiments\manual_search\` folder.

The script requires pytorch. 

The GPT model has been adapted from Andrej Karpathy's excellent [Neural networks zero to hero lectures.](https://github.com/karpathy/nn-zero-to-hero)
