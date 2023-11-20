from model import sk_model
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_train_datasets, load_data, make_dataset, get_test_data
from model import SetConv
from data import encode_samples, encode_data, normalize_labels

def sk_train(model_num, train_x, train_y, test_x):
    model = sk_model(model_num)
    model.fit(train_x, train_y)
    predicts = model.predict(test_x)
    result = []

    if model_num == 0:
        for i in range(len(predicts)):
            result[i] = abs(int(predicts[i][0]))        
    else:
        for i in range(len(predicts)):
            result[i] = abs(int(predicts[i]))
    
    return result

def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype = np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype = np.int64)

def error_loss(preds, targets, min_val, max_val):
    error = []
    preds = torch.exp((preds * (max_val - min_val)) + min_val)
    targets = torch.exp((targets * (max_val - min_val)) + min_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            error.append(preds[i] / targets[i])
        else:
            error.append(targets[i] / preds[i])
    return torch.mean(torch.cat(error))


def predict(model, data_loader, cuda):
    preds = []

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, targets = \
                samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = \
                sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()

        samples, predicates, joins, targets = \
            Variable(samples), Variable(predicates), Variable(joins), Variable(targets)
        
        sample_masks, predicate_masks, join_masks = \
            Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])
    return preds

def network_train(train_file_path, test_file_path, num_queries, num_epochs, batch_size, hid_units, cuda):
    # Load training and validation data
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size = batch_size)
    test_data_loader = DataLoader(test_data, batch_size = batch_size)

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(targets)
            sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = error_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

    # Get final training and validation set predictions
    preds_train = predict(model, train_data_loader, cuda)
    preds_test = predict(model, test_data_loader, cuda)

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Load test data
    joins, predicates, tables, samples, label = load_data(test_file_path, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size = batch_size)

    preds_test = predict(model, test_data_loader, cuda)
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    return preds_test_unnorm