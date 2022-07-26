from binance.client import Client
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime
from zigzag_functions import *
from patterns import *
import os
from depth_patterns import *

OPEN, HIGH, LOW, CLOSE, VOLUME = 'open', 'high', 'low', 'close', 'volume'
NUM_OF_TRADES = 'num_of_trades'
OHLC4 = 'ohlc4'
VOL_PRICE_DIV_NOT = 'vol_price_not'
PERIOD, COIN, INTERVAL = 'period', 'coin', 'interval'



# bottom of the code shoud be edited

def create_df(data, zigzag_accuracy, interval):
    df = pd.DataFrame(data, columns=['t', 'open', 'high', 'low', 'close', 'volume', 'time', 'quote_avolume',
                                     'num_of_trades', 'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'ignore'])
    df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'num_of_trades']]
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df.set_index(np.arange(len(df)), inplace=True)
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['hlcc4'] = (df['high'] + df['low'] + df['close'] + df['close']) / 4
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['C-O/O'] = (df['close'].div(df['open']) - 1) * 100
    df['H-L/L'] = (df['high'].div(df['low']) - 1) * 100
    df['Co/HL'] = df['C-O/O'].div(df['H-L/L'])
    df = calculate_mix_zigzag(df, zigzag_accuracy, interval)
    return df


def plot_chart(df, file_name):
    df.index = pd.DatetimeIndex(df['time'].map(
        lambda x: datetime.utcfromtimestamp(int(x) / 1000).strftime('%Y-%m-%d %H:%M:%S')))
    save = dict(fname=file_name,
                dpi=300,)
    mpf.plot(df, type='candle', style='binance', savefig=save)


def calculate_descriptions(df, number_of_chunks):
    df_len = len(df)
    C_O_desc = pd.DataFrame(columns=[
                            'start_time', 'end_time', 'count', 'mean', 'std', 'min', 'max', '50%', '25%', '75%'])
    H_L_desc = pd.DataFrame(columns=[
                            'start_time', 'end_time', 'count', 'mean', 'std', 'min', 'max', '50%', '25%', '75%'])
    Co_HL_desc = pd.DataFrame(columns=[
                              'start_time', 'end_time', 'count', 'mean', 'std', 'min', 'max', '50%', '25%', '75%'])
    start = str(datetime.fromtimestamp(df.iloc[0]['time']/1000))[:19]
    end = str(datetime.fromtimestamp(df['time'][df_len-1]/1000))[:19]
    c_o_desc_temp_full = df['C-O/O'].describe()
    c_o_desc_temp_full['start_time'] = start
    c_o_desc_temp_full['end_time'] = end
    C_O_desc = C_O_desc.append(c_o_desc_temp_full, ignore_index=True)
    h_l_desc_temp_full = df['H-L/L'].describe()
    h_l_desc_temp_full['start_time'] = start
    h_l_desc_temp_full['end_time'] = end
    H_L_desc = H_L_desc.append(h_l_desc_temp_full, ignore_index=True)
    co_hl_desc_temp_full = df['Co/HL'].describe()
    co_hl_desc_temp_full['start_time'] = start
    co_hl_desc_temp_full['end_time'] = end
    Co_HL_desc = Co_HL_desc.append(co_hl_desc_temp_full, ignore_index=True)

    chunk_len = int(df_len/number_of_chunks)
    for i in range(number_of_chunks):
        start = i*chunk_len
        end = (i+1)*chunk_len
        temp_df = df.loc[start:end]
        temp_df_len = len(temp_df)
        start = str(datetime.fromtimestamp(temp_df['time'].iat[0]/1000))[:19]
        end = str(datetime.fromtimestamp(
            temp_df['time'].iat[temp_df_len-1]/1000))[:19]
        c_o_desc_temp = temp_df['C-O/O'].describe()
        c_o_desc_temp['start_time'] = start
        c_o_desc_temp['end_time'] = end
        C_O_desc = C_O_desc.append(c_o_desc_temp, ignore_index=True)
        h_l_desc_temp = temp_df['H-L/L'].describe()
        h_l_desc_temp['start_time'] = start
        h_l_desc_temp['end_time'] = end
        H_L_desc = H_L_desc.append(h_l_desc_temp, ignore_index=True)
        co_hl_desc_temp = temp_df['Co/HL'].describe()
        co_hl_desc_temp['start_time'] = start
        co_hl_desc_temp['end_time'] = end
        Co_HL_desc = Co_HL_desc.append(co_hl_desc_temp, ignore_index=True)
    return C_O_desc, H_L_desc, Co_HL_desc


def delete_outlier_rows(df, source, number_of_parts, accuracy):
    parts_len = len(df)//number_of_parts
    result_df = pd.DataFrame(columns=df.columns)
    for i in range(number_of_parts + 1):
        start = i*parts_len
        temp_df = df.iloc[start:start+parts_len]
        if len(temp_df) == 0:
            break
        avg = temp_df[source].mean()
        temp_df = temp_df[(temp_df[source]/avg > 1-accuracy)
                          & (temp_df[source]/avg < 1+accuracy)]
        result_df = result_df.append(temp_df, ignore_index=True)
    return result_df


def calculate_patterns_for_all_chunks(df: pd.DataFrame, depth):
    result_df = pd.DataFrame(columns=[
                             'pattern', 'count', 'vol_avg', 'n_o_t_avg', 'highest_lowest_avg', 'price*vol/not'])
    df['index_diff'] = df.index
    df['index_diff'] = df['index_diff'].diff()
    splitter_df = df[df['index_diff'] != 1]
    if splitter_df.shape[0] == 0:
        return result_df
    st_time = splitter_df['time'].iat[0]
    for index, row in splitter_df.iterrows():
        end_time = row['time']
        # print('end_idx=',end_time)
        temp_df = df[(df['time'] >= st_time) & (df['time'] < end_time)]
        if len(temp_df) > 1:
            res = find_patterns_with_depth_(temp_df, depth)
            print('res_len=', res.shape)
            result_df = pd.concat([result_df, res], axis=0)
        st_time = end_time
    if result_df.shape[0] == 0:
        return result_df
    result = result_df.drop_duplicates('pattern')
    print(result_df.shape, '_______________________________________')
    result['count'] = result.apply(
        lambda row: result_df[result_df.pattern == row.pattern]['count'].sum(), axis=1)
    result = result.sort_values(by=['count'], ascending=False)
    result['percent'] = result['count'] * 100 / result['count'].sum()
    return result


def filter_df_and_calculate_patterns(df, source, depth):
    print(f'calculating filtered df for {source} with depth {depth}')
    gt_df = df[df[source] > 0]
    lt_df = df[df[source] < 0]

    gt_res = calculate_patterns_for_all_chunks(gt_df, depth)
    lt_res = calculate_patterns_for_all_chunks(lt_df, depth)
    return gt_res, lt_res


def plot_dict_of_patterns_df(dfs: dict, depth):
    for key, df in dfs.items():
        plot_depth_pattern_graph(df=df, depth=depth, fname=key)


def plot_dict_of_paired_patterns_df(dfs: dict, depth):
    for key, df in dfs.items():
        plot_paired_depth_pattern_graph(
            df=df, depth=depth, fname=key+'_paired')


def calculate_avg_stdev_difference(df, source, stdev_multiplier):
    col_name = f'avg_stdev{stdev_multiplier}_difference'
    number = df[source].mean() - np.std(df[source])*stdev_multiplier
    df[col_name] = df[source] - number
    gt_df_percent = df[df[f'avg_stdev{stdev_multiplier}_difference']
                       > 0].shape[0] * 100 / df.shape[0]
    lt_df_percent = df[df[f'avg_stdev{stdev_multiplier}_difference']
                       < 0].shape[0] * 100 / df.shape[0]
    return col_name, gt_df_percent, lt_df_percent


def calculate_avg_stdev_difference(df, source, stdev_multiplier):
    col_name = f'avg_stdev{stdev_multiplier}_{source}_difference'
    number = df[source].mean() + np.std(df[source])*stdev_multiplier
    df[col_name] = df[source] - number
    return df


def calculate_add_avg_stdev(df, sources_list, stdev_multiplier_list):
    for stdev_multiplier in stdev_multiplier_list:
        for source in sources_list:
            df = calculate_avg_stdev_difference(df, source, stdev_multiplier)
    return df


def get_difference(df, source):
    df[OPEN] = df[OPEN].diff()
    df[HIGH] = df[HIGH].diff()
    df[LOW] = df[LOW].diff()
    df[CLOSE] = df[CLOSE].diff()
    df = df.iloc[1:]
    df['sign'] = df[source].apply(lambda x: 1 if x > 0 else -1)
    return df


def save_result_dict_to_excel(result: dict, fname):
    def excel_columns(col):
        """ Convert given row and column number to an Excel-style col name. """
        LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        result = []
        while col:
            col, rem = divmod(col - 1, 26)
            result[:0] = LETTERS[rem]
        return ''.join(result)

    def get_df_col_widths(df: pd.DataFrame):
        """ Return Excel-style column widths for a given dataframe. """
        idx_max = max([len(str(s))
                      for s in df.index.values] + [len(str(df.index.name))])
        return [idx_max] + [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]

    with pd.ExcelWriter(fname, engine='xlsxwriter') as writer:
        for name, df in result.items():
            df.to_excel(writer, sheet_name=name)
            worksheet = writer.sheets[name]
            y, x = df.shape
            y += 1  # 1 for the header row
            x += 1  # 1 for the index and column
            worksheet.conditional_format(
                f'C2:C{y}', {'type': 'data_bar', 'bar_solid': True})  # count column
            for i in range(4, x + 1):
                worksheet.conditional_format(f'{excel_columns(i)}2:{excel_columns(i)}{y}', {'type': '3_color_scale',
                                                                                            'mid_color': '#00000000'})  # other every column
            for i, width in enumerate(get_df_col_widths(df)):  # set columns auto fit
                worksheet.set_column(i, i, width)
            # Freeze first row and first 2 columns.
            worksheet.freeze_panes(1, 2)
        print(f"{fname} created")


def main(symbol, interval, start_time, end_time, number_of_chunks, diff, diff_source, zigzag_accuracy, patterns_depth, plot=True):
    start_time = datetime.strptime(
        start_time, "%Y-%m-%d %H:%M:%S").timestamp()*1000
    end_time = datetime.strptime(
        end_time, "%Y-%m-%d %H:%M:%S").timestamp()*1000

    s_fname = f'{symbol.lower()}_spot_{interval}.xlsx'
    f_fname = symbol.lower() + '_perp_' + interval + '.xlsx'
    if os.path.exists(s_fname):
        s_df = pd.read_excel(s_fname)
        s_df.set_index(np.arange(len(s_df)), inplace=True)
    else:
        client = Client()
        print('Getting spot klines for symbol:', symbol,
              'from', start_time, 'to', end_time)
        klines = client.get_historical_klines(
            symbol, interval, int(start_time), int(end_time))
        s_df = create_df(klines, zigzag_accuracy, interval)
        s_df.to_excel(s_fname)

    if os.path.exists(f_fname):
        f_df = pd.read_excel(f_fname)
        f_df.set_index(np.arange(len(f_df)), inplace=True)
    else:
        client = Client()
        print('Getting perp klines for symbol:', symbol,)
        klines = client.futures_historical_klines(
            symbol, interval, int(start_time), int(end_time))
        f_df = create_df(klines, zigzag_accuracy, interval)
        f_df.to_excel(f_fname)

    if diff:
        print('Calculating difference')
        s_df = get_difference(s_df, diff_source)
        f_df = get_difference(f_df, diff_source)
    else:
        s_df['sign'] = s_df.apply(
            lambda x: 1 if x[OPEN] < x[CLOSE] else -1, axis=1)
        f_df['sign'] = f_df.apply(
            lambda x: 1 if x[OPEN] < x[CLOSE] else -1, axis=1)

    s_df[VOL_PRICE_DIV_NOT] = s_df[VOLUME] * s_df[CLOSE] / s_df[NUM_OF_TRADES]
    f_df[VOL_PRICE_DIV_NOT] = f_df[VOLUME] * f_df[CLOSE] / f_df[NUM_OF_TRADES]
    spot_descriptions = calculate_descriptions(s_df, number_of_chunks)
    perp_descriptions = calculate_descriptions(f_df, number_of_chunks)

    if plot:
        print('Plotting klines')
        plot_chart(s_df, symbol + '_' + interval + '_' + 'spot.png')
        plot_chart(f_df, symbol + '_' + interval + '_' + 'perp.png')

    STDEV_MULTIPLIERS = [-2, -1, 0, 1, 2]
    AVG_STDEV_SOURCES = [VOLUME, NUM_OF_TRADES, VOL_PRICE_DIV_NOT]
    s_df = calculate_add_avg_stdev(s_df, AVG_STDEV_SOURCES, STDEV_MULTIPLIERS)
    # f_df = calculate_add_avg_stdev(f_df, AVG_STDEV_SOURCES, STDEV_MULTIPLIERS)

    filtered_dfs_dict = {}
    print('calculating filtered dfs')
    for source in AVG_STDEV_SOURCES:
        for stdev_multiplier in STDEV_MULTIPLIERS:
            avg_stdev_src = f'avg_stdev{stdev_multiplier}_{source}_difference'
            print('source', source)
            result = filter_df_and_calculate_patterns(
                s_df, avg_stdev_src, patterns_depth)
            sheet_key = f'avg-sdv{stdev_multiplier}_{source}_dif'
            filtered_dfs_dict[sheet_key + '_gt'] = result[0]
            filtered_dfs_dict[sheet_key + '_lt'] = result[1]
    save_result_dict_to_excel(
        filtered_dfs_dict, symbol + interval + 'avg_stdevs.xlsx')
    plot_dict_of_patterns_df(filtered_dfs_dict, depth=patterns_depth)
    plot_dict_of_paired_patterns_df(filtered_dfs_dict, depth=patterns_depth//2)

    # zigzag_patterns = calculate_zigzag_patterns(df, 'zigzag')
    print('Finding Patterns')
    spot_pos_patterns, spot_neg_patterns, spot_pos_patterns_chunked, spot_neg_patterns_chunked = calculate_chunked_patterns(
        s_df, 'open', 'close', number_of_chunks)
    plot_positive_tree_graph(
        spot_pos_patterns, fname=symbol + '_' + interval + '_' + 'spot_pos_patterns.png')
    plot_negative_tree_graph(
        spot_neg_patterns, fname=symbol + '_' + interval + '_' + 'spot_neg_patterns.png')
    perp_pos_patterns, perp_neg_patterns, perp_pos_patterns_chunked, perp_neg_patterns_chunked = calculate_chunked_patterns(
        f_df, 'open', 'close', number_of_chunks)
    plot_positive_tree_graph(
        perp_pos_patterns, fname=symbol + '_' + interval + '_' + 'perp_pos_patterns.png')
    plot_negative_tree_graph(
        perp_neg_patterns, fname=symbol + '_' + interval + '_' + 'perp_neg_patterns.png')

    # patterns with depth _____________________________
    spot_depth_patterns = find_patterns_with_depth(s_df, depth=10)
    plot_depth_pattern_graph(
        spot_depth_patterns, fname=symbol + '_' + interval + '_spot', depth=10)
    perp_depth_patterns = find_patterns_with_depth(f_df, depth=7)
    plot_depth_pattern_graph(
        perp_depth_patterns, fname=symbol + '_' + interval + '_perp', depth=7)
    plot_paired_depth_pattern_graph(
        spot_depth_patterns, fname=symbol + '_' + interval + '_spot_paired_depth', depth=5)
    plot_paired_depth_pattern_graph(
        perp_depth_patterns, fname=symbol + '_' + interval + '_perp_paired_depth', depth=5)
    print('Saving dataframe to csv')
    writer = pd.ExcelWriter(symbol + '_' + interval + '_' +
                            str(start_time) + '_' + str(end_time) + '.xlsx', engine='xlsxwriter')


# C_O_desc, H_L_desc, Co_HL_desc
    spot_descriptions[0].to_excel(writer, sheet_name='C_O_desc')
    perp_descriptions[0].to_excel(writer, sheet_name='C_O_desc', startcol=12)
    spot_descriptions[1].to_excel(writer, sheet_name='H_L_desc')
    perp_descriptions[1].to_excel(writer, sheet_name='H_L_desc', startcol=12)
    spot_descriptions[2].to_excel(writer, sheet_name='Co_HL_desc')
    perp_descriptions[2].to_excel(writer, sheet_name='Co_HL_desc', startcol=12)
    spot_pos_patterns.to_excel(writer, sheet_name='Patterns', index=False)
    spot_neg_patterns.to_excel(
        writer, sheet_name='Patterns', startcol=9, index=False)
    perp_pos_patterns.to_excel(
        writer, sheet_name='Patterns', startcol=18, index=False)
    perp_neg_patterns.to_excel(
        writer, sheet_name='Patterns', startcol=27, index=False)
    spot_pos_patterns_chunked.to_excel(
        writer, sheet_name='Chunked_patterns_pos', index=False)
    perp_pos_patterns_chunked.to_excel(
        writer, sheet_name='Chunked_patterns_pos', startcol=10, index=False)
    spot_neg_patterns_chunked.to_excel(
        writer, sheet_name='Chunked_patterns_neg', index=False)
    perp_neg_patterns_chunked.to_excel(
        writer, sheet_name='Chunked_patterns_neg', startcol=10, index=False)
    spot_depth_patterns.to_excel(
        writer, sheet_name='Depth_patterns', index=False)
    perp_depth_patterns.to_excel(
        writer, sheet_name='Depth_patterns', startcol=12, index=False)
    writer.save()


# ______________________Edit_here________________________________
# symbol = 'BTCUSDT'
# interval = '5m'
# # format: YYYY-MM-DD HH:MM:SS
# start_time = '2020-01-01 00:00:00'
# end_time = '2021-01-01 00:00:00'
# number_of_chunks = 12 # split data to chunks and analyze each chunk
# diff = False
# diff_source = 'high'
# patterns_depth = 8 # depth of calculated patterns
# main(symbol, interval, start_time,
#      end_time, number_of_chunks, diff, diff_source, zigzag_accuracy=0.013, patterns_depth=patterns_depth, plot=False)
   