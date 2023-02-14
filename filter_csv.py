import init_paths
from config import cfg
from config import update_config
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import os 


def RSSINormalize(df, ap):
    rssimax = df['RSSI'].max()
    rssimin = df['RSSI'].min()
    print(f'Sniffer {ap+1} min is {rssimin}, max is {rssimax}')
    df['RSSI'] = ( df['RSSI'] - rssimin ) / ( rssimax - rssimin )
    
    return df

def RSSINormalizeForTest(cfg, df, ap):
    rssimax = cfg.TEST.RSSI_LIST[ap][0]
    rssimin = cfg.TEST.RSSI_LIST[ap][1]
    #print(f'Sniffer {ap+1} min is {rssimin}, max is {rssimax}')
    df['RSSI'] = ( df['RSSI'] - rssimin ) / ( rssimax - rssimin )
    
    return df
    
def GetData(cfg, is_train=False):
    try :
        if len(cfg.START_TIME) != len(cfg.END_TIME) :
            raise ValueError('Error : Start time list length is not equal to End time list')

        # save each dataframe of sniffers
        before_list = []
        for id in range(cfg.AP_NUMS):
            input_path = os.path.join(cfg.BEFORE_DIR + cfg.FILE + cfg.AP_NAME[id] +'.'+ cfg.FILE_TYPE)
            df = pd.read_csv(input_path)
            before_list.append(df)

        for t in range(len(cfg.START_TIME)):
            block_count = 0
            output_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(cfg.LABEL_LIST[t]+1)  + '.' + cfg.FILE_TYPE)

            if os.path.isfile(output_path):
                os.remove(output_path)

            mergedata = DataFilter(cfg, before_list[0], t, 0, is_train)
            mergedata = GetMeanData(mergedata)
            
            #mergedata.columns = columns
            for j in range(1, cfg.AP_NUMS): 
                # how = 'outer' means A union B
                to_merge = DataFilter(cfg, before_list[j], t, j, is_train)
                to_merge = GetMeanData(to_merge)

                mergedata = pd.merge(mergedata, to_merge, on='time', how='outer')
                mergedata = mergedata.sort_values(by='time')

            # new column names
            columns = ['time']
            for id in range(cfg.AP_NUMS):
                columns.append(cfg.FILE + cfg.AP_NAME[id])
            
            mergedata.columns = columns

            # change type from object to float
            mergedata[columns[1:]] = mergedata[columns[1:]].astype(str).astype(float)
            # set limit_direction as 'both' to avoid first or last value is NaN 
            mergedata[columns[1:]] = mergedata[columns[1:]].interpolate(axis=0, limit_direction='both')

            mergedata['label'] = int(cfg.LABEL_LIST[t])
            block_count += len(mergedata)
            print(f'Block {t+1} length is : {block_count}')
            mergedata.to_csv(output_path, mode='a', header=not os.path.isfile(output_path), date_format='%H:%M:%S', index = False)

    except ValueError as msg:
        print(msg)
 
def GetPhoneData(cfg, is_train=False):
    try :
        if len(cfg.START_TIME) != len(cfg.END_TIME) :
            raise ValueError('Error : Start time list length is not equal to End time list')

        # save each dataframe of sniffers
        before_list = []
        for id in range(cfg.AP_NUMS):
            input_path = os.path.join(cfg.BEFORE_DIR + cfg.FILE + cfg.AP_NAME[id] +'.'+ cfg.FILE_TYPE)
            df = pd.read_csv(input_path)
            before_list.append(df)

        # generate final results
        for t in range(len(cfg.START_TIME)):
            block_count = 0
            output_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(t+1)  + '.' + cfg.FILE_TYPE)

            if os.path.isfile(output_path):
                os.remove(output_path)

            mergedata = PhoneDataFilter(cfg, before_list[0], t, 0, is_train)

            #mergedata.columns = columns
            for j in range(1, cfg.AP_NUMS): 
                # how = 'outer' means A union B
                mergedata = pd.merge(mergedata, PhoneDataFilter(cfg, before_list[j], t, j, is_train), on='Timestamp', how='outer')
                mergedata = mergedata.sort_values(by='Timestamp')

            # new column names
            columns = ['time']
            for id in range(cfg.AP_NUMS):
                columns.append(cfg.FILE + cfg.AP_NAME[id])
            
            mergedata.columns = columns

            # change type from object to float
            mergedata[columns[1:]] = mergedata[columns[1:]].astype(str).astype(float)
            # set limit_direction as 'both' to avoid first or last value is NaN 
            mergedata[columns[1:]] = mergedata[columns[1:]].interpolate(axis=0, limit_direction='both')

            mergedata['label'] = int(cfg.LABEL_LIST[t])
            block_count += len(mergedata)
            print(f'Block {t+1} length is : {block_count}')
            mergedata.to_csv(output_path, mode='a', header=not os.path.isfile(output_path), date_format='%H:%M:%S', index = False)

    except ValueError as msg:
        print(msg)

def DataFilter(cfg, df, t, ap, is_train):
    df = df.loc[(df.uuid == cfg.UUID)]
    df = df[['time', 'RSSI']]
    # remove decimal point part
    df['time'] = df['time'].str.split('.', expand=True)[0]
    # filter value in specific time range
    timeindex = pd.DatetimeIndex(df['time'])
    if is_train:
        df = RSSINormalize(df, ap)
    else:
        df = RSSINormalizeForTest(cfg, df, ap)
    df = df.iloc[timeindex.indexer_between_time(cfg.START_TIME[t], cfg.END_TIME[t])]
    return df

# use mobile phone as a sniffer
def PhoneDataFilter(cfg, df, t, ap, is_train):
    df = df.loc[(df['Device Name'] == cfg.DEVICE_NAME)]
    df = df[['Timestamp', 'RSSI']]
    df['Timestamp'] = [new_t[:-4] for new_t in df['Timestamp']]
    # filter value in specific time range
    timeindex = pd.DatetimeIndex(df['Timestamp'])
    if is_train:
        df = RSSINormalize(df, ap)
    else:
        df = RSSINormalizeForTest(cfg, df, ap)
    df = df.iloc[timeindex.indexer_between_time(cfg.START_TIME[t], cfg.END_TIME[t])]
    return df

def DataFilterNoLabel(cfg, df, ap, is_train):
    df = df.loc[(df.uuid == cfg.UUID)]
    df = df[['time', 'RSSI']]
    # remove decimal point part
    df['time'] = df['time'].str.split('.', expand=True)[0]
    # filter value in specific time range
    timeindex = pd.DatetimeIndex(df['time'])
    if is_train:
        df = RSSINormalize(df, ap)
    else:
        df = RSSINormalizeForTest(cfg, df, ap)
    df = df.iloc[timeindex.indexer_between_time(cfg.START_TIME[0], cfg.END_TIME[-1])]
    return df

# get mean 
def GetMeanData(df):
    # save new result
    new_df = pd.DataFrame()
    # get unique time index
    uni_index = df['time'].unique()
    #print(uni_index)

    for i, time in enumerate(uni_index):
        #print(df[df['time'] == uni_index[i]]['RSSI'].mean(axis=0))
        new_df = new_df.append({'time': time, 'RSSI': df[df['time'] == uni_index[i]]['RSSI'].mean(axis=0)}, ignore_index=True)
    return new_df


def GetTestData(cfg):
    try :
        if len(cfg.START_TIME) != len(cfg.END_TIME) :
            raise ValueError('Error : Start time list length is not equal to End time list')

        # save each dataframe of sniffers
        before_list = []
        for id in range(cfg.AP_NUMS):
            input_path = os.path.join(cfg.BEFORE_DIR + cfg.FILE + cfg.AP_NAME[id] +'.'+ cfg.FILE_TYPE)
            df = pd.read_csv(input_path)
            before_list.append(df)


        output_path = os.path.join(cfg.AFTER_DIR + 'test' + '.' + cfg.FILE_TYPE)

        if os.path.isfile(output_path):
            os.remove(output_path)

        total_count = 0
        use_testing2 = False
        for t in range(len(cfg.START_TIME)):

            mergedata = DataFilter(cfg, before_list[0], t, 0, False)
            mergedata = GetMeanData(mergedata)
            # reverse when using testing 2
            mergedata = mergedata.sort_values(['time'], ascending= not use_testing2)

            for j in range(1, cfg.AP_NUMS): 
                # how = 'outer' means A union B
                to_merge = DataFilter(cfg, before_list[j], t, j, False)
                to_merge = GetMeanData(to_merge)
                # reverse when using testing 2
                to_merge = to_merge.sort_values(['time'], ascending= not use_testing2)
                mergedata = pd.merge(mergedata, to_merge, on='time', how='outer')
                # ascending is False when using testing 2
                mergedata = mergedata.sort_values(by='time', ascending= not use_testing2)
                
            #print(mergedata)
            # new column names
            columns = ['time']
            for id in range(cfg.AP_NUMS):
                columns.append(cfg.FILE + cfg.AP_NAME[id])
            
            mergedata.columns = columns

            # change type from object to float
            #print(mergedata[columns[1:]].astype(str))
            mergedata[columns[1:]] = mergedata[columns[1:]].astype(str).astype(float)
            # set limit_direction as 'both' to avoid first or last value is NaN 
            mergedata[columns[1:]] = mergedata[columns[1:]].interpolate(axis=0, limit_direction='both')


            mergedata['label'] = int(cfg.LABEL_LIST[t])
            total_count += len(mergedata)
            mergedata.to_csv(output_path, mode='a', header=not os.path.isfile(output_path), date_format='%H:%M:%S', index = False)

        df = pd.read_csv(output_path)
        for block in range(len(np.unique(cfg.LABEL_LIST))):
            block_ = df[df['label'] == block]
            print(f"Block {block} :")
            for i, name in enumerate(cfg.AP_NAME):
                print(f"Sniffer {i} rssi mean : {block_[cfg.FILE + name].mean()}")


    except ValueError as msg:
        print(msg)


def GetPhoneTestData(cfg):
    try :
        if len(cfg.START_TIME) != len(cfg.END_TIME) :
            raise ValueError('Error : Start time list length is not equal to End time list')

        # save each dataframe of sniffers
        before_list = []
        for id in range(cfg.AP_NUMS):
            input_path = os.path.join(cfg.BEFORE_DIR + cfg.FILE + cfg.AP_NAME[id] +'.'+ cfg.FILE_TYPE)
            df = pd.read_csv(input_path)
            before_list.append(df)


        output_path = os.path.join(cfg.AFTER_DIR + 'test' + '.' + cfg.FILE_TYPE)

        if os.path.isfile(output_path):
            os.remove(output_path)

        total_count = 0

        for t in range(len(cfg.START_TIME)):

            mergedata = DataFilter(cfg, before_list[0], t, 0, False)
            for j in range(1, cfg.AP_NUMS): 
                # how = 'outer' means A union B
                df2 = DataFilter(cfg, before_list[j], t, j, False)
                mergedata = pd.merge(mergedata, df2, on='time', how='outer')
                mergedata = mergedata.sort_values(by='time')
                
            #print(mergedata)
            # new column names
            columns = ['time']
            for id in range(cfg.AP_NUMS):
                columns.append(cfg.FILE + cfg.AP_NAME[id])
            
            mergedata.columns = columns

            # change type from object to float
            #print(mergedata[columns[1:]].astype(str))
            mergedata[columns[1:]] = mergedata[columns[1:]].astype(str).astype(float)
            # set limit_direction as 'both' to avoid first or last value is NaN 
            mergedata[columns[1:]] = mergedata[columns[1:]].interpolate(axis=0, limit_direction='both')


            mergedata['label'] = int(cfg.LABEL_LIST[t])
            total_count += len(mergedata)
            mergedata.to_csv(output_path, mode='a', header=not os.path.isfile(output_path), date_format='%H:%M:%S', index = False)


    except ValueError as msg:
        print(msg)

def GetTestDataNoLabel(cfg):
    try :
        if len(cfg.START_TIME) != len(cfg.END_TIME) :
            raise ValueError('Error : Start time list length is not equal to End time list')

        # save each dataframe of sniffers
        before_list = []
        for id in range(cfg.AP_NUMS):
            input_path = os.path.join(cfg.BEFORE_DIR + cfg.FILE + cfg.AP_NAME[id] +'.'+ cfg.FILE_TYPE)
            df = pd.read_csv(input_path)
            before_list.append(df)


        output_path = os.path.join(cfg.AFTER_DIR + 'test' + '.' + cfg.FILE_TYPE)

        if os.path.isfile(output_path):
            os.remove(output_path)

        total_count = 0

        mergedata = DataFilterNoLabel(cfg, before_list[0], 0, False)
        mergedata = GetMeanData(mergedata)
        mergedata = mergedata.sort_values(['time'])

        for j in range(1, cfg.AP_NUMS): 
            # how = 'outer' means A union B
            to_merge = DataFilterNoLabel(cfg, before_list[j], j, False)
            to_merge = GetMeanData(to_merge)

            mergedata = pd.merge(mergedata, to_merge, on='time', how='outer')
            mergedata = mergedata.sort_values(by='time')
            
        #print(mergedata)
        # new column names
        columns = ['time']
        for id in range(cfg.AP_NUMS):
            columns.append(cfg.FILE + cfg.AP_NAME[id])
        
        mergedata.columns = columns

        # change type from object to float
        #print(mergedata[columns[1:]].astype(str))
        mergedata[columns[1:]] = mergedata[columns[1:]].astype(str).astype(float)
        # set limit_direction as 'both' to avoid first or last value is NaN 
        mergedata[columns[1:]] = mergedata[columns[1:]].interpolate(axis=0, limit_direction='both')


        total_count += len(mergedata)
        mergedata.to_csv(output_path, mode='a', header=not os.path.isfile(output_path), date_format='%H:%M:%S', index = False)

        df = pd.read_csv(output_path)

    except ValueError as msg:
        print(msg)



def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the csv and label')
    # general

    parser.add_argument('--cfg',
                        help='Configure file of the csv ',
                        required=True,
                        type=str)
    parser.add_argument('--type',
                        help='Sniffer is Rasberry Pi or Phone',
                        required=True,
                        default='pi',
                        type=str)
    # optional
                        
    parser.add_argument('--beforeDir',
                        help='File path of the csv before preprocessing',
                        type=str,
                        default='')

    parser.add_argument('--afterDir',
                        help='File path of the csv after preprocessing',
                        type=str,
                        default='')

    parser.add_argument('--mode',
                        help='Mode about how to get data and do normalization, train or valid or test',
                        type=str,
                        default='train')
    args = parser.parse_args()
    return args

def main():
    arg = parse_args()
    # update cfg with yaml file
    update_config(cfg, arg)

    output_dir = Path(cfg.AFTER_DIR)
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir()
    # filtering data (rasberry pi or phone)
    if arg.type == "pi":
        # for training and validation
        if arg.mode != "test":
            GetData(cfg,"train" == arg.mode)
        # Generate test.csv for testing
        elif not cfg.TEST_NO_LABEL:
            GetTestData(cfg)
        else:
            GetTestDataNoLabel(cfg)
    elif arg.type == "phone":
        # for training and validation
        if arg.mode != "test":
            GetPhoneData(cfg,"train" == arg.mode)
        # Generate test.csv for testing
        else:
            GetPhoneTestData(cfg)

if __name__ == '__main__':
    main()