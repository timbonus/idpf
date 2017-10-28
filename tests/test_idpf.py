import numpy as np
import pandas as pd
import datetime
from keras.models import Sequential
from keras.layers import Dense
#import keras.backend as K
import idpf
from pytest import raises

# fix random seed for reproducibility
np.random.seed(7)

train_frame = pd.read_pickle('ts_month_train.pickle')
test_frame = pd.read_pickle('ts_month_test.pickle')

def test_idpf_time_series_check():
    assert idpf._is_forecast_ready(train_frame)
    with raises(ValueError, message='Time series index is not '+\
              'chronologically ordered.'):
        idpf._is_forecast_ready(train_frame.sort_index(ascending=False))
    with raises(TypeError, message='Accepted time series input is '+\
              'pandas DataFrame object with DatetimeIndex.'):
        idpf._is_forecast_ready(train_frame.values)

def test_idpf_label_separator():
    train_row, train_col = train_frame.shape
    X_train, Y_train = idpf._extract_labels(train_frame,label_count=1)
    assert len(X_train.columns) == train_col - 1
    assert len(X_train) == len(Y_train)
    assert len(X_train) == train_row
    assert not len(X_train) == train_row + 1
    
def test_idpf_fixed_shift_label_extractor():
    shift = 5
    column_to_drop = train_frame.columns[-1]
    train_frame_label_drop = train_frame.drop(column_to_drop,axis=1)
    train_row, train_col = train_frame_label_drop.shape
    train_frame_shift = idpf._generate_shifted_label(
                                train_frame_label_drop,shift=shift)
    X_train, Y_train = idpf._extract_labels(train_frame_shift,label_count=1)
    assert train_frame.iloc[10,-1] == X_train.iloc[10,0]
    assert train_frame.iloc[10,-1] != X_train.iloc[10,-1]
    assert len(X_train.columns) == train_col
    assert len(X_train) == len(Y_train)
    assert len(X_train) == train_row - shift
    
def test_idpf_multiple_shift_label_extractor():
    shift_range = range(1,5)
    column_to_drop = train_frame.columns[-1]
    train_frame_label_drop = train_frame.drop(column_to_drop,axis=1)
    train_row, train_col = train_frame_label_drop.shape
    train_frame_shifts = idpf._generate_multiple_shifted_labels(
                                train_frame_label_drop,shifts=shift_range)
    X_train, Y_train = idpf._extract_labels(train_frame_shifts,label_count=4)
    assert len(X_train.columns) == train_col
    assert len(X_train) == len(Y_train)
    assert len(X_train) == train_row - sum(shift_range)
    

def test_idpf_model_labeller():
    reindexed_frame = idpf._time_based_indexer(train_frame)
    assert reindexed_frame.index.levels[0][0] == datetime.time(0, 0)

def test_idpf_generate_base_mlp_model():
    model = idpf.create_mlp_model(layer_node_counts=[4,8,2])
    assert isinstance(model,Sequential) 
    
    model = idpf.create_mlp_model(layer_node_counts=[5,8,1])
    assert isinstance(model,Sequential)
    

def test_idpf_generate_time_base_models():
    train_frame_time = idpf._time_based_indexer(train_frame)
    model_dict = idpf._generate_time_base_mlp_models(train_frame,[4,8,2])
    np.testing.assert_array_equal(list(model_dict.keys()),
                                  train_frame_time.index.levels[0])
    models = list(model_dict.values())
    for ob in models: assert isinstance(ob,Sequential)

def test_idpf_fit_time_specific_model():
    model_dict = idpf._generate_time_base_models(train_frame,[5,8,1])
    history = idpf._fit_time_specific_model(
                    datetime.time(1,0),model_dict,train_frame)
    
    assert isinstance(history.model,Sequential)
    
def test_idpf_create_all_mlp_models():    
    with raises(ValueError, message='Input dataframe column count must '+
                                    'match the sum of input and output layers '+
                                    'node count.'):
        assert idpf.create_time_based_mlp_models(train_frame,
                                                 layer_node_counts=[1,1,1])
                                                 
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[5,8,1],
                                                   batch_size=100,
                                                   epochs=1)
    for model in model_dict:
        assert model_dict[model].count_params() == 57
 
def test_idpf_predict_time_specific_model():
    # Testing single output ANN.
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[5,8,1],
                                                   batch_size=100,
                                                   epochs=1)
    
    test_output = idpf._predict_time_specific_model(datetime.time(0,0),
                                                    model_dict,test_frame)
    X_test, Y_test = idpf._extract_labels(test_frame,label_count=1)
    assert test_output['prediction'].shape == \
            pd.DataFrame(Y_test.loc[datetime.time(0,0)]).shape
    np.testing.assert_array_equal(test_output['prediction'].index,
                          test_frame.loc[datetime.time(0,0)].index)
    assert isinstance(test_output['loss'], float)
    
    # Testing multiple output ANN.    
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[4,8,2],
                                                   batch_size=100,
                                                   epochs=1)
    
    test_output = idpf._predict_time_specific_model(datetime.time(0,0),
                                                    model_dict,test_frame)
    X_test, Y_test = idpf._extract_labels(test_frame,label_count=2)
    assert test_output['prediction'].shape == \
            pd.DataFrame(Y_test.loc[datetime.time(0,0)]).shape
    np.testing.assert_array_equal(test_output['prediction'].index,
                          test_frame.loc[datetime.time(0,0)].index)
    assert isinstance(test_output['loss'], float)
 
def test_idpf_validate_single_multiple_input_models_with_test_data():
    X_test, Y_test = idpf._extract_labels(test_frame,label_count=1)
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[5,8,1])
    with raises(ValueError, message='Input dataframe column count must '+
                                    'match the sum of input and output layers '+
                                    'node counts of the models provided in '+\
                                    'model_dict.'):
        assert idpf.test_validation(test_frame.iloc[:,1:],
                                    model_dict)
    # Testing single output ANN.
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[5,8,1],
                                                   batch_size=100,
                                                   epochs=1)
    pred_frame, loss_frame = idpf.test_validation(test_frame,model_dict)
    
    assert isinstance(pred_frame,pd.DataFrame)
    assert pred_frame.shape == Y_test.shape
    assert pred_frame.index.is_monotonic_increasing
    
    assert isinstance(loss_frame,pd.Series)
    assert loss_frame.dtype.type == np.float64
    
    # Testing multiple output ANN.
    X_test, Y_test = idpf._extract_labels(test_frame,label_count=2)
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[4,8,2],
                                                   batch_size=100,
                                                   epochs=1)
    pred_frame, loss_frame = idpf.test_validation(test_frame,model_dict)
    
    assert isinstance(pred_frame,pd.DataFrame)    
    assert pred_frame.shape == Y_test.shape
    assert pred_frame.index.is_monotonic_increasing
    
    assert isinstance(loss_frame,pd.Series)
    assert loss_frame.dtype.type == np.float64
    
def test_idpf_point_predict_shape():
    model_dict = idpf.create_time_based_mlp_models(train_frame,
                                                   layer_node_counts=[4,8,2],
                                                   batch_size=100,
                                                   epochs=1)
                                                   
    X_test, Y_test = idpf._extract_labels(test_frame,label_count=2)
    X_test_point = X_test.iloc[10:11,:]
    Y_test_point = Y_test.iloc[10:11,:]
    
    X_predict_point = idpf.point_predict(X_test_point,model_dict)
    
    assert X_predict_point.shape == Y_test_point.shape    
    