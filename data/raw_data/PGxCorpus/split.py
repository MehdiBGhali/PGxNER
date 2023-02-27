import argparse, os, sys, re
from sklearn.model_selection import train_test_split

def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    parser.add_argument("--test", default=0.1, type=float)
    parser.add_argument("--dev", default=0.1, type=float)
    parser.add_argument("--output_dir", default="./split", type=str)
    parser.add_argument("--input_text", default="./text", type=str)

    args, _ = parser.parse_known_args()
    return args

def get_split_ids(ids,train,dev,test): 
    train_ids, int_ids = train_test_split(ids, train_size = train, test_size = dev+test)
    dev_ids, test_ids = train_test_split(int_ids, train_size = dev, test_size = test)
    return train_ids, dev_ids, test_ids

if __name__ == "__main__":
    args = parse_parameters()
    ids = [document.replace(".txt", "") for document in os.listdir(args.input_text)]
    train_ratio, dev_ratio, test_ratio = 1-(args.test+args.dev), args.dev, args.test
    train_ids, dev_ids, test_ids = get_split_ids(ids, train_ratio, dev_ratio, test_ratio)
    with open(os.path.join(args.output_dir,"train.id"),"w") as f : 
        for document in train_ids : 
            f.write(f"{document}\n")
    with open(os.path.join(args.output_dir,"dev.id"),"w") as f : 
        for document in dev_ids : 
            f.write(f"{document}\n")
    with open(os.path.join(args.output_dir,"test.id"),"w") as f : 
        for document in test_ids : 
            f.write(f"{document}\n")