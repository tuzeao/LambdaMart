from lambdamart import LambdaMART
import numpy as np


def get_data(file_loc):
    f = open(file_loc, 'r')
    data = []
    for line in f:
        new_arr = []
        arr = line.split(' #')[0].split()
        ''' Get the score and query id '''
        score = arr[0]
        q_id = arr[1].split(':')[1]
        new_arr.append(int(score))
        new_arr.append(int(q_id))
        arr = arr[2:]
        ''' Extract each feature from the feature vector '''
        for el in arr:
            new_arr.append(float(el.split(':')[1]))
        data.append(new_arr)
    f.close()
    return np.array(data)


def main():
    training_data = get_data("example_data/train.txt")
    test_data = get_data("example_data/test.txt")
    model = LambdaMART(training_data=training_data,
                       number_of_trees=2,
                       learning_rate=0.1)
    model.fit()

    average_ndcg, predicted_scores = model.validate(test_data, k=2)
    predicted_scores = model.predict(test_data[:, 1:])

    print(average_ndcg)
    print("#"*30)
    print(predicted_scores)
    model.save('example_data/example_model')


if __name__ == '__main__':
    main()
