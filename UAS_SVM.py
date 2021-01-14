#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dir_X_train = 'Train/X_train.txt'
dir_y_train = 'Train/y_train.txt'

dir_X_test = 'Test/X_test.txt'
dir_y_test = 'Test/y_test.txt'


# In[ ]:


import numpy

features = numpy.loadtxt('features.txt', str)
slc_ft = ['tBodyAcc-STD-1', 'tBodyAcc-STD-2', 'tBodyAcc-STD-3', 'tBodyAcc-SMA-1', 'tGravityAcc-Mean-1', 'tGravityAcc-Mean-2', 'tGravityAcc-Mean-3',]

n = len(features)
slc_ft_idx = []
for ft in slc_ft:
    i = 0
    for j in range(i, n):
        if ft == features[j]:
            slc_ft_idx.append(j)
            i = j + 1
            break
            
print(slc_ft_idx)


# In[ ]:


X_train = numpy.loadtxt(dir_X_train, usecols=slc_ft_idx)
y_train = numpy.loadtxt(dir_y_train)

X_test = numpy.loadtxt(dir_X_test, usecols=slc_ft_idx)
y_test = numpy.loadtxt(dir_y_test)


# In[ ]:


print(len(y_test))


# In[ ]:


print(len(y_train))
print(len(y_test))
print(len(X_train[0]))


# In[ ]:


labels = numpy.loadtxt('activity_labels.txt', str)
#print(labels)


# In[ ]:


activity_labels = [i[1] for i in labels]
#print(activity_labels)


# In[ ]:


transition_labels = [activity_labels[i] for i in range(len(activity_labels)) if i >= 6]
walking_labels = [activity_labels[i] for i in range(len(activity_labels)) if i < 3]

summary_labels = [[walking_labels, 'WALKING'], 'SITTING', 'STANDING', 'LAYING', [transition_labels, 'TRANSITION']]


# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics

def fit_model(kernel, degree):
    model = SVC(C=1.0, kernel=kernel, degree=degree, gamma='scale')
    model.fit(X_train, y_train)

    expected = y_test
    predicted = model.predict(X_test)
    
    report = metrics.classification_report(expected, predicted, target_names=activity_labels)
    report_dict = metrics.classification_report(expected, predicted, target_names=activity_labels, output_dict=True)
    conf_matrix = metrics.confusion_matrix(expected, predicted)
    
    return [report, report_dict, conf_matrix]


# In[ ]:


def label_avg(label, row_name, report_dict):
    if isinstance(label, str):
        return report_dict[label][row_name]
    
    sumPrec = 0
    for l in label:
        sumPrec += report_dict[l][row_name]

    row = sumPrec / len(label)
    return row


# In[ ]:


def print_summary(summary, row_name, report_dict):
    for c in summary:
        if isinstance(c, list):
            v = label_avg(c[0], row_name, report_dict)
            print(c[1], '-->', '{:.3f}%'.format(v * 100))
        else:
            v = label_avg(c, row_name, report_dict)
            print(c, '-->', '{:.3f}%'.format(v * 100))


# In[ ]:


test_config = [(kernel, degree) for kernel in ['linear', 'poly', 'rbf', 'sigmoid'] for degree in range(1,5)]
print(test_config)


# In[ ]:


test_config_2 = [('poly', degree) for degree in range(1,11)]
print(test_config_2)


# In[ ]:


for kernel, degree in test_config:
    report, report_dict, conf_mat = fit_model(kernel, degree)
    print(kernel.upper(), degree)
    print_summary(summary_labels, 'precision', report_dict)
    print('-' * 30)


# In[ ]:


for kernel, degree in test_config_2:
    report, report_dict, conf_mat = fit_model(kernel, degree)
    print(kernel.upper(), degree)
    print_summary(summary_labels, 'precision', report_dict)
    print('-' * 30)

