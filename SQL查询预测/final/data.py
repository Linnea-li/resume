import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_train_data(file_path, split):
    train_data = get_data(file_path, True)
    column_details, table_index, feature_index = get_column_details()

    if split == True:
        train_data, test_data = train_test_split(train_data, test_size = 0.1, random_state=42)
        train_data_x = []
        train_data_y = []
        for i in range(len(train_data)):
            train_data_x.append(train_data[i][0:3])
            train_data_y.append(train_data[i][3])
        
        train_data_x = np.array(create_X_vector(train_data_x, feature_index, column_details))
        scaler = StandardScaler().fit(train_data_x)
        train_data_x = scaler.transform(train_data_x)

        train_data_y = np.array(train_data_y)
        train_data_y = train_data_y[:, np.newaxis] 

        test_data_x = []
        test_data_y = []
        for i in range(len(test_data)):
            test_data_x.append(test_data[i][0:3])
            test_data_y.append(test_data[i][3])
        
        test_data_x = np.array(create_X_vector(test_data_x, feature_index, column_details))
        test_data_x = scaler.transform(test_data_x)

        test_data_y = np.array(test_data_y)
        test_data_y = test_data_y[:, np.newaxis]      

        return train_data_x, train_data_y, test_data_x, test_data_y, scaler

    else:
        train_data_x = []
        train_data_y = []
        for i in range(len(train_data)):
            train_data_x.append(train_data[i][0:3])
            train_data_y.append(train_data[i][3])
        
        train_data_x = np.array(create_X_vector(train_data_x, feature_index, column_details))
        scaler = StandardScaler().fit(train_data_x)
        train_data_x = scaler.transform(train_data_x)

        train_data_y = np.array(train_data_y)
        train_data_y = train_data_y[:, np.newaxis] 

        return train_data_x, train_data_y, scaler


def get_test_data(file_path, scaler):
    test_data = get_data(file_path, False)
    column_details, table_index, feature_index = get_column_details()
    test_x = np.array(create_X_vector(test_data, feature_index, column_details))
    test_x = scaler.transform(test_x)
    return test_x


def get_column_details():
    column_details = pd.read_csv(r"C:\Users\74285\Desktop\大数据大作业\ruccardinality\column_min_max_vals.csv")
    table_index = {'t' : 0, 'mc' : 1, 'ci' : 2, 'mi' : 3, 'mi_idx' : 4, 'mk' : 5}
    feature_index = {}
    for i in range(len(column_details)):
        feature_index[column_details.iloc[i]['name']] = i
    
    return column_details, table_index, feature_index

def get_data(file_path, train):
    csvfile = open(file_path, 'r')
    data = [each for each in csv.reader(csvfile, delimiter = '#')]

    for i in range(len(data)):
        tables = data[i][0].split(',')
        for j in range(len(tables)):
            tables[j] = tables[j].split(' ')[1]
        data[i][0] = tables 

        joins = []
        if len(data[i][0]) > 1:
            join_cons = data[i][1].split(',')
            for j in range(len(join_cons)):
                join_cons[j] = join_cons[j].split('=')
                joins.append(join_cons[j])
        data[i][1] = joins

        items = data[i][2].split(',')
        conditions = []
        for j in range(int(len(items) / 3)):
            condition = []
            condition.append(items[3 * j])
            condition.append(items[3 * j + 1])
            condition.append(int(items[3 * j + 2])) 
            conditions.append(condition)
        data[i][2] = conditions

        if train == True:
            data[i][3] = int(data[i][3])
    return data

def create_X_vector(example, feature_index, column_details):
    X = []
    for i in range (len(example)):
        str1 = example[i][1]
        str2 = example[i][2]
        X_temp = np.zeros(60)

        for j in range(len(str1)):
            temp1 = str1[j]       
            for k in range(len(temp1)):
                temp_name = temp1[k]                     
                num = feature_index.get(temp_name)
                X_temp[num + 40] = 1

        for k in range(len(str2)):
            temp = str2[k]
            if temp[1] == '=':
                temp_name = temp[0]
                num = feature_index.get(temp_name)
                X_temp[num * 2] = temp[2]
                X_temp[num * 2 + 1] = temp[2]
            elif temp[1] == '>':
                temp_name = temp[0]
                num = feature_index.get(temp_name)
                X_temp[num * 2] = temp[2]
                X_temp[num * 2 + 1] = column_details.iloc[num]['max']            
            else :
                temp_name = temp[0]
                num = feature_index.get(temp_name)
                X_temp[num * 2 + 1] = temp[2]
                X_temp[num * 2] = column_details.iloc[num]['min']
        X.append(X_temp)
    return X

def normalize_labels(labels):
    labels = np.array([np.log(float(l)) for l in labels])
    min_val = labels.min()
    max_val = labels.max()
    labels_norm = (labels - min_val) / (max_val - min_val)
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

def load_and_encode_train_data(num_queries, num_materialized_samples):
    train_data_x, train_data_y, test_data_x, test_data_y, scaler = get_train_data("training_data.csv", True)
    column_details, table_index, feature_index = get_column_details() 
    samples_enc = create_X_vector(train_data_x, feature_index, column_details)
    label_norm, min_val, max_val = normalize_labels(train_data_y)
    dicts = [column_details, table_index, feature_index]
    return dicts, min_val, max_val, train_data_y, test_data_y, train_data_x, test_data_x


def make_dataset(samples, predicates, joins, labels, max_num_joins, max_num_predicates):
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks,
                                 predicate_masks, join_masks)

def get_train_datasets(num_queries, num_materialized_samples):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = \
        load_and_encode_train_data(num_queries, num_materialized_samples)
    train_dataset = make_dataset(*train_data, labels = labels_train, max_num_joins = max_num_joins,
                                 max_num_predicates = max_num_predicates)
    test_dataset = make_dataset(*test_data, labels = labels_test, max_num_joins = max_num_joins,
                                max_num_predicates = max_num_predicates)
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset


def write_data(result, file_path):
    with open(file_path, 'w', newline = "") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Query ID", "Predicted Cardinality"])
        for i in range(len(result)):
            writer.writerow([i, result[i]])