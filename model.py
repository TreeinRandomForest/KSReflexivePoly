import numpy as np
import copy
from keras.models import Sequential
from keras.layers import Dense

def clean_d3_data(filename, zeropad = False):
    '''
    '''
    with open(filename) as f:
        features = []
        target = []

        n_examples = 0
        n_max = 0
        for line in f:
            if line.find(":")>-1:
                line_split = line.rstrip("\n").split()

                if n_examples > 0:
                    features.append(np.array(matrix))
                    target.append(pic)
                
                d = int(line_split[0])
                n = int(line_split[1])
                pic = int(line_split[-2].split(":")[1])

                if n_max < n:
                    n_max = n
                
                if d != 3:
                    raise ValueError()
                
                n_examples += 1
                
                matrix = [] #store 3xN matrix of polytope lattice points
            else:
                coordinates = [int(val) for val in line.split()]

                if len(coordinates) > 0:
                    matrix.append(np.array(coordinates))

        #process last example
        features.append(np.array(matrix))
        target.append(pic)

        target = np.array(target)

        #zero-pad
        if zeropad:
            features = [np.hstack([f, np.zeros((3, n_max - f.shape[1]))]) for f in features]
        features = np.array(features)
        
        return(features, target)

def augment_data(features, target, seed=0, n_transforms=5):
    np.random.seed(seed)

    augmented_features = []
    augmented_target = []

    example_index = []
    
    for f_index in range(len(features)):
        f = features[f_index]
        t = target[f_index]
        
        for n in range(n_transforms): #generate permutations
            f_augment = f.transpose()[np.random.permutation(f.shape[1]),:].transpose() #permute the columns i.e. the lattice points

            augmented_features.append(f_augment)
            augmented_target.append(t)

            example_index.append(f_index)
            
    augmented_features, augmented_target = np.array(augmented_features), np.array(augmented_target)

    #labels for augmented (1) vs original data (0)
    augment_labels = np.concatenate([np.zeros(len(features)), np.ones(len(augmented_features))], axis=0)
    
    features = np.append(features, augmented_features, axis=0)
    target = np.append(target, augmented_target, axis=0)

    example_index = np.concatenate([np.arange(len(features)), np.array(example_index)], axis=0)
    
    return(features, target, augment_labels, example_index)

def flatten_data(features):
    flattened_features = []
    for f in features:
        flattened_features.append(np.reshape(f, f.shape[0]*f.shape[1]))
    flattened_features = np.array(flattened_features)

    return(flattened_features)
        
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(units = 1000, activation='sigmoid', input_dim=input_dim))
    model.add(Dense(units = 100, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return(model)

def train_test_split(features, target, augment_labels = None, example_index=None, train_size=0.5):
    N = len(features)

    indices = list(range(N))
    np.random.shuffle(indices)

    N_train = int(train_size * N)

    train_indices, test_indices = indices[0:N_train], indices[N_train:]

    train_features, test_features = features[train_indices], features[test_indices]
    train_target, test_target = target[train_indices], target[test_indices]

    result = {'train_features': train_features,
            'train_target': train_target,
            'test_features': test_features,
            'test_target': test_target
    }

    if augment_labels is not None:
        train_augment_labels, test_augment_labels = augment_labels[train_indices], augment_labels[test_indices]
        result['train_augment_labels'] = train_augment_labels
        result['test_augment_labels'] = test_augment_labels
    if example_index is not None:
        train_example_index, test_example_index = example_index[train_indices], example_index[test_indices]
        result['train_example_index'] = train_example_index
        result['test_example_index'] = test_example_index
        
    return(result)

def binarize_target(target, threshold):
    target = (target > threshold).astype(int)

    return target

def train_model(model, train_features, train_target, epochs = 100):
    model.fit(train_features, train_target, epochs = epochs, batch_size=64)

    return(model)

def evaluate_model(model, features_target_dict):
    pass

if __name__ == "__main__":
    threshold = 10
    filename = "data/d3/RefPoly.d3"
    augment = True
    epochs = 40
    n_transforms = 5
    
    features, target = clean_d3_data(filename, zeropad = True) #read and clean data

    if augment: #permutations of vertices
        features, target, augment_labels, example_index = augment_data(features, target, seed=0, n_transforms=n_transforms)

    features = flatten_data(features) #prepare features
    target = binarize_target(target, threshold) #prepare target

    #Train test split
    if augment:
        d = train_test_split(features, target, augment_labels=augment_labels, example_index=example_index, train_size=0.5)
    else:
        d = train_test_split(features, target, augment_labels=None, example_index=None, train_size=0.5)

    #build and train model
    model = build_model(len(features[0]))
    model = train_model(model, d['train_features'], d['train_target'], epochs = epochs)

    #model validation
    train_pred = model.predict(d['train_features'])
    test_pred = model.predict(d['test_features'])

    if augment:
        train_original_pred = train_pred[d['train_augment_labels']==0]
        test_original_pred = test_pred[d['test_augment_labels']==0]

        train_original_pred = [int(i[0] > 0.5) for i in train_original_pred]
        test_original_pred = [int(i[0] > 0.5) for i in test_original_pred]
        
        train_original_target = d['train_target'][d['train_augment_labels']==0]
        test_original_target = d['test_target'][d['test_augment_labels']==0]
        
