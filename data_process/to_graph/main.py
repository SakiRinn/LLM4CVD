import os
import os.path as osp
import argparse
import json

from gensim.models import Word2Vec

from graphs import json_to_graphs
from tokenizer import code_tokenize


def train_test_split(data, train_ratio=0.8, val_ratio=0.1):
    train_split_index = int(len(data) * train_ratio)
    val_split_index = int(len(data) * (train_ratio + val_ratio))

    train_dataset = data[:train_split_index]
    val_dataset = data[train_split_index:val_split_index]
    test_dataset = data[val_split_index:]

    return {'train_GGNNinput': train_dataset,
            'valid_GGNNinput': val_dataset,
            'test_GGNNinput': test_dataset}

def train_w2v(sentences, epochs=5, min_count=1, embedding_size=128,
              output_dir='outputs/w2v/'):
    words = []
    for sentence in sentences:
        words.append(code_tokenize(sentence))

    print(f'Total words: {len(words)}')
    print('Training...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    w2v_model = Word2Vec(words, vector_size=embedding_size, min_count=min_count, workers=8)
    for i in range(epochs):
        w2v_model.train(words, total_examples=len(words), epochs=1)
        w2v_model.save(os.path.join(output_dir, f'e{i+1}.bin'))

    print('Completed!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='The .json file of raw data.')
    parser.add_argument('--w2v', default='', help='The .bin file of Word2vec model.')
    parser.add_argument('--output-dir', help='The directory to the output all files.', default='outputs/graph')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not args.w2v:
        print("Train a new w2v model...")
        w2v_dir = osp.join(args.output_dir, osp.basename(args.data_path).split(".")[0], 'w2v')
        with open(args.data_path, 'r') as f:
            data = json.load(f)
            sentences = [e['code'] for e in data]
        train_w2v(sentences, output_dir=w2v_dir)
        args.w2v = osp.join(w2v_dir, 'e5.bin')
        print("Training completed!")

    w2v_model = Word2Vec.load(args.w2v)
    print("Success to load w2v model, start processing...")

    graphs = json_to_graphs(w2v_model, args.data_path, args.output_dir)
    graph_datasets = train_test_split(graphs)

    for k, v in graph_datasets.items():
        with open(os.path.join(args.output_dir, osp.basename(args.data_path).split(".")[0], f'{k}.json'), 'w') as f:
            json.dump(v, f)
    print('Completed!')


if __name__ == '__main__':
    main()
