

from datasets import load_dataset,DatasetDict
for dataset in ['emotion', 'SetFit/sst5', 'rotten_tomatoes', 'hate']:
    dataname = dataset
    if dataname in ['emoji', "sentiment", "stance_abortion", "stance_atheism", "stance_climate", "stance_feminist", \
                "stance_hillary", 'hate', 'emotion']:
        dataset = load_dataset('tweet_eval', dataname)
    else:
        dataset = load_dataset(dataname)

    # read data from dataset
    df_trian = dataset['train'].to_pandas()
    df_trian = df_trian[df_trian['label'].apply(lambda x: x in [0,1])]
    df_trian['exp_split'] = "train"
    df_test = dataset['test'].to_pandas()
    df_test = df_test[df_test['label'].apply(lambda x: x in [0,1])]
    df_test['exp_split'] = "test"
    df = df_trian.append(df_test)
    df = df.dropna(axis=0,how='any')
    
    print('token mean length:',df['text'].apply(lambda x: len(x)).mean())
    print("train data negative:",df_trian['label'].count()-df_trian['label'].sum())
    print("train data positive:",df_trian['label'].sum())
    print("test data negative:",df_test['label'].count()-df_test['label'].sum())
    print("test data positive:",df_test['label'].sum())

    if dataname == "SetFit/sst5":
        dataname = "sst"

    import os
    df_path = f'./preprocess/{dataname}'
    try:
        os.mkdir(df_path)
    except:
        pass

    df_file = f"./preprocess/{dataname}/data.csv"
    df.to_csv(df_file)