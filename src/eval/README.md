# Evaluation 

Models that are trained with the original regressor can also be evaluated using the toes regressors only in inference mode!


### Requirements
- Python > 3.6
- Datasets in TF Record Format 
    - if you haven't generated the tf records so far see [datasets_preprocessing/README.md](../../datasets_preprocessing/README.md)


### Evaluation

1. install requirements
    - (optional) setup new virtual environment  
        ```
        mkvirtualenv hmr2-eval
        workon hmr2-eval
        pip install -U pip
        ```
    - install requirements 
        ```
        pip install -r eval/requirements.txt
        ```
2. check `eval_config`
    - change `data_dir` to your path
    - add or remove entries given the available models
3. check `evaluation.py`
    - change paths in class `EvalConfig`
4. run evaluation
    ```
    python evaluation.py
    ```
