import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
#import keras.backend as K

'''
Provides methods for fitting an ensemble of time-of-day-specific models for 
load profiles with daily seasonality. 

Training and test data can be utilised to measure model performance using the
create_time_based_mlp_models and test_validation methods.

Point prediction for forecasting can be performed using point_predict method.
'''

def _is_forecast_ready(dataframe):
    '''
    Validates input is pandas DataFrame with chronologically ordered 
    DatetimeIndex and provides informative exceptions if not.
    
    TODO: Validate time series indexed with a regular period that is a 
    standard fraction of an hour - e.g. 5, 10, 15, 30, 60 min.
    
    Inputs:
    dataframe - a pandas dataframe. 
    
    Returns:
    boolean - True if valid input.
    
    Raises:
    Exceptions if input is not valid.
    '''
    try: 
        if dataframe.index.is_monotonic_increasing:
            return True
        else:
            raise ValueError('Time series index is not chronologically ordered.')
    except AttributeError:
        raise TypeError('Accepted time series input is pandas DataFrame '+\
              'object with DatetimeIndex.')
        
def _extract_labels(dataframe,label_count):
    '''
    This function removes the first column from the passed data frame for use as
    the time series to be forecasted.
    
    Inputs:
    dataframe - a pandas dataframe. 
    
    Returns:
    Y_dataframe (pandas.Series) - the first column of dataframe as a pandas 
    series.
    X_dataframe (pandas.DataFrame) - the remaining columns of dataframe.
    '''
    
    X_dataframe = dataframe.iloc[:,label_count:]
    Y_dataframe = dataframe.iloc[:,:label_count]
    
    return X_dataframe, Y_dataframe
    
def _generate_shifted_label(dataframe,shift):
    '''
    This function assumes the last column is the raw time series and makes a 
    copy of it to use as training label. It is appended at the front of the 
    dataframe.
    
    Inputs:
    dataframe - a pandas dataframe. 
    
    Returns:
    labelled_dataframe (pandas.DataFrame) - the new dataframe with an additional label
    column appended to the front.
    '''
    
    label_col = pd.DataFrame(dataframe.iloc[:,-1]).shift(shift)
    label_col.columns = [str(shift)+' steps ahead']
    labelled_df = pd.concat([label_col,dataframe],axis=1)
    labelled_df.dropna(inplace=True)
    return labelled_df
    
def _generate_multiple_shifted_labels(dataframe,shifts):
    for shift in shifts:
        dataframe = _generate_shifted_label(dataframe,shift)
    return dataframe

def create_mlp_model(layer_node_counts,
                      activation='relu',
                      loss='mean_squared_error',
                      optimizer='adam'):
    '''
    Creates a multilayer perceptron.
    '''
    model = Sequential()
    input_layer = None
    for i,layer in enumerate(layer_node_counts):
        if i == 0:
            input_layer = layer
        elif i == 1:
            model.add(Dense(layer, input_dim=input_layer, activation=activation))
        else:
            model.add(Dense(layer, activation=activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def _time_based_indexer(dataframe):
    '''
    Reindexes DatetimeIndex to have time of day / date heirarchical MultiIndex.
    '''
    dataframe = dataframe.copy()
    dataframe.index = [dataframe.index.time,dataframe.index.date]
    return dataframe
    
def _generate_time_base_models(dataframe,
                               layer_node_counts,
                               activation='elu',
                               loss='mean_squared_error',
                               optimizer='adam'):
    '''
    Creates a dictionary of models. Key is discrete times of day in the dataset.
    Each key points to distinct model object, each of which can be tuned 
    separately.
    '''
    time_frame = _time_based_indexer(dataframe)
    
    model_dict = {time: create_mlp_model(layer_node_counts=layer_node_counts,
                                         activation=activation,
                                         loss=loss,
                                         optimizer=optimizer) 
                               for time in time_frame.index.levels[0]}
    return model_dict
    
def _fit_time_specific_model(time,model_dict,dataframe,batch_size=1,epochs=1):
    '''
    Allows a time index to be specified and the associated model has training 
    data to be passed to it via its fit method.
    '''
    model = model_dict[time]
    data = dataframe.loc[time]
    X_data, Y_data = _extract_labels(data,model.output_shape[1])
    return model.fit(X_data.values, 
                     Y_data.values,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=0)
    
def create_time_based_mlp_models(dataframe,
                                 layer_node_counts,
                                 activation='relu',
                                 batch_size=1,
                                 epochs=1):
    '''
    This needs to match up layer node counts with dataframe.
    '''
    _is_forecast_ready(dataframe) 
    
    input_nodes = layer_node_counts[1]
    output_nodes = layer_node_counts[-1]
    if not (layer_node_counts[0] + layer_node_counts[-1]) == len(dataframe.columns):
        raise ValueError('Input dataframe column count must match the sum of '+\
                         'input and output layers node counts.')
    
    model_dict = _generate_time_base_models(dataframe, 
                                            layer_node_counts,
                                            activation=activation)
    
    for time in model_dict:
        _fit_time_specific_model(time,
                                 model_dict,
                                 dataframe,
                                 batch_size=batch_size,
                                 epochs=epochs)
    
    return model_dict

def _predict_time_specific_model(time,model_dict,dataframe):
    model = model_dict[time]
    data = dataframe.loc[time]
    X_data, Y_data = _extract_labels(data,model.output_shape[1])
    model_evaluation = model.evaluate(X_data.values, Y_data.values)
    model_output = pd.DataFrame(model.predict(X_data.values),index=Y_data.index)
    
    return {'loss': model_evaluation, 'prediction': model_output}
    
def test_validation(dataframe,model_dict):
    '''
    This is where you know the output.
    '''
    model_sample = list(model_dict.values())[0]
    
    input_nodes = int(model_sample.input.shape[1])
    output_nodes = int(model_sample.output.shape[1])
    
    if not (input_nodes + output_nodes) == len(dataframe.columns):
        raise ValueError('Input dataframe column count must match the sum of '+\
                         'input and output layers node counts of the models '+\
                         'provided in model_dict.')
    
    test_output = {time: _predict_time_specific_model(time,model_dict,dataframe)
                                                        for time in model_dict}
    
    time_index = list(test_output.keys())
    
    loss_frame = pd.Series([test_output[time]['loss'] for time in time_index],
                                    index = time_index)
    loss_frame.sort_index(inplace=True)
    
    pred_frame = pd.concat([test_output[time]['prediction'] for time in time_index])
    pred_frame.sort_index(inplace=True)
    
    return pred_frame, loss_frame

def point_predict(dataframe,model_dict):
    model_sample = list(model_dict.values())[0]
    
    input_nodes = int(model_sample.input.shape[1])
    
    if not input_nodes == len(dataframe.columns):
        raise ValueError('Input dataframe column count must equal the number '+\
                         'of inputs to the model.') 

    if not dataframe.shape[0] == 1:
        raise ValueError('For point prediction input dataframe row count '+\
                         'must have only one row.')
    
    try:
        time = dataframe.index.time[0]
    except:
        raise TypeError('Point prediction requires input with DatetimeIndex.')
    
    model = model_dict[time]
    
    return pd.DataFrame(model.predict(dataframe.values),index=dataframe.index)
                                
