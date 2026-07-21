# COSI 115b Final Project

All package requirements are in `requirements.txt`.

To improve Hugging Face download speed and in order to use Gemma-4-31b to generate hard negative samples, a `.env` file should be created to put the `HF_TOKEN` and `AI_STUDIO_API_KEY` environment variables, in the format `ENV_VAR=value`.

To train the model, run `train()` in `train.py` with a chosen checkpoint from `model/` or no checkpoint to train from scratch.

To code to evaluate the model can be found in the `__main__` section of `evaluate.py`.