import os
import time
import json
import pickle as pkl
import numpy as np
import io
from io import BytesIO
import xgboost as xgb
import pandas as pd
import sagemaker_xgboost_container.encoder as xgb_encoders
NUM_FEATURES = 58


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    print("--------------- model_fn Start ---------------")
    model_file = "xgboost-model"
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, model_file))
    print("--------------- model_fn End ---------------")
    return model
                     

def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).
    """
    print("--------------- input_fn Start ---------------")
    print("Content type: ", request_content_type)
    if request_content_type == "text/csv":
        test_df = pd.read_csv(io.StringIO(request_body), header=0)
        print("--------------- input_fn End ---------------")
        return xgb.DMatrix(test_df)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )
        

def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array (predictions and scores)
    """
    print("--------------- predict_fn Start ---------------")
    start_time = time.time()
    y_probs = model.predict(input_data)
    print("--- Inference time: %s secs ---" % (time.time() - start_time))    
    y_preds = [1 if e >= 0.5 else 0 for e in y_probs] 
    #feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    print("--------------- predict_fn End ---------------")
    return np.vstack((y_preds, y_probs))


def output_fn(predictions, content_type="application/json"):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    print("--------------- output_fn Start ---------------")
    if content_type == "text/csv":
        return ','.join(str(x) for x in outputs)
    elif content_type == "application/json":
        outputs = json.dumps({
            'pred': predictions[0,:].tolist(),
            'prob': predictions[1,:].tolist()
        })        
        print("--------------- output_fn End ---------------")
        return outputs
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
