import pandas as pd
from datetime import datetime
from igraph import *


"""
functions to find sign candle patterns:
subpatterns are patterns inside a pattern, 
for example '++--' contains '++-', '+--', '+-',
these are calculated when subpatterns selected,
positive and negative indicates this:
positive: pattern starts when we see a '+' after a '-'
negative: pattern starts when we see a '-' after a '+'
"""



def find_positive_patterns_with_sub_patterns(df, open_source, close_source):
    print('Finding candle patterns')
    df_len = len(df)
    st__time = str(datetime.fromtimestamp(df['time'].iat[0]/1000))[:19]
    vol_dict = {}
    pattern_list = []
    pattern = [0, 0]
    patt_temp = 1
    vol = 0
    n_o_t = 0
    hst = 0
    lst = 100000
    positive_vol = []
    negative_vol = []
    positive_not = []
    negative_not = []
    positive_highest_ls = []
    positive_lowest_ls = []
    negative_highest_ls = []
    negative_lowest_ls = []
    negative_close_ls = []

    def add_to_pattern_dict(pattern, vol, current_n_o_t, hst_lst, current_close):
        pattern_str = str(pattern[0]) + '-' + str(pattern[1])
        pattern_list.append(pattern)
        if pattern_str in vol_dict:
            vol_dict[pattern_str][0] += vol
            vol_dict[pattern_str][1] += current_n_o_t
            vol_dict[pattern_str][2].append(hst_lst)
            vol_dict[pattern_str][3].append(current_close)
        else:
            vol_dict[pattern_str] = [
                vol, current_n_o_t, [hst_lst], [current_close]]
    for i in range(df_len):
        current_patt = 1 if (df[close_source].iat[i] -
                             df[open_source].iat[i]) >= 0 else -1
        current_vol = df['volume'].iat[i]
        current_n_o_t = df['num_of_trades'].iat[i]
        current_close = df['close'].iat[i]
        current_high = df['high'].iat[i]
        current_low = df['low'].iat[i]
        if patt_temp == 1 and current_patt == -1:
            pattern[1] += 1
            positive_vol.append(current_vol)
            positive_not.append(current_n_o_t)
            negative_highest_ls.append(current_high)
            negative_lowest_ls.append(current_low)
            negative_close_ls.append(current_close)
        elif patt_temp == -1 and current_patt == -1:
            pattern[1] += 1
            negative_vol.append(current_vol)
            negative_not.append(current_n_o_t)
            negative_highest_ls.append(current_high)
            negative_lowest_ls.append(current_low)
            negative_close_ls.append(current_close)
        elif patt_temp == 1 and current_patt == 1:
            pattern[0] += 1
            positive_vol.append(current_vol)
            positive_not.append(current_n_o_t)
            positive_highest_ls.append(current_high)
            positive_lowest_ls.append(current_low)
        elif patt_temp == -1 and current_patt == 1:
            positive_vol = positive_vol[::-1]
            positive_not = positive_not[::-1]
            positive_highest_ls = positive_highest_ls[::-1]
            positive_lowest_ls = positive_lowest_ls[::-1]
            for j in range(1, pattern[0]+1):
                for k in range(1, pattern[1]+1):
                    patt = [j, k]
                    vol = sum(positive_vol[:j]+negative_vol[:k])
                    n_o_t = sum(positive_not[:j]+negative_not[:k])
                    hst = max(positive_highest_ls[:j]+negative_highest_ls[:k])
                    lst = min(positive_lowest_ls[:j]+negative_lowest_ls[:k])
                    current_close = negative_close_ls[k-1]
                    add_to_pattern_dict(
                        patt, vol, n_o_t, hst-lst, current_close)
            pattern = [1, 0]
            vol = 0
            n_o_t = 0
            hst, lst = 0, 100000
            positive_vol = []
            negative_vol = []
            positive_not = []
            negative_not = []
            positive_highest_ls = []
            positive_lowest_ls = []
            negative_highest_ls = []
            negative_lowest_ls = []
            negative_close_ls = []
        patt_temp = current_patt

    patterns = []
    for patt in pattern_list:
        if patt in patterns:
            continue
        patterns.append(patt)
    for pattern in patterns:
        count = pattern_list.count(pattern)
        patt_from_dict = vol_dict[str(pattern[0]) + '-' + str(pattern[1])]
        vol_avg = patt_from_dict[0] / (count*sum(pattern))
        n_o_t_avg = patt_from_dict[1] / (count*sum(pattern))
        highest_lowest_avg = sum(patt_from_dict[2]) / len(patt_from_dict[2])
        avg_price = sum(patt_from_dict[3]) / len(patt_from_dict[3])
        price_vol_not = (avg_price * vol_avg) / n_o_t_avg if n_o_t_avg else 0
        pattern.extend(
            [vol_avg, n_o_t_avg, highest_lowest_avg, price_vol_not, count])
    pattern_df = pd.DataFrame(patterns, columns=[
                              '+', '-', 'vol_avg', 'n_o_t_avg', 'highest_lowest_avg', 'price*vol/not', 'count'])
    pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    pattern_df['percent'] = (pattern_df['count'] /
                             pattern_df['count'].sum()) * 100
    pattern_df['time'] = st__time
    pattern_df = pattern_df[['time', '+', '-', 'vol_avg', 'n_o_t_avg',
                             'highest_lowest_avg', 'price*vol/not', 'count', 'percent']]
    return pattern_df


def find_positive_patterns(df):
    print('Finding candle patterns')
    df_len = len(df)
    vol_dict = {}
    st_time = str(datetime.fromtimestamp(df['time'].iat[0]/1000))[:19]
    pattern_list = []
    pattern = [0, 0]
    patt_temp = 1
    vol = 0
    n_o_t = 0
    higest_list = []
    lowest_list = []


    def add_to_pattern_dict(pattern, vol, current_n_o_t, hst_lst, current_close):
        pattern_str = str(pattern[0]) + '-' + str(pattern[1])
        pattern_list.append(pattern)
        if pattern_str in vol_dict:
            vol_dict[pattern_str][0] += vol
            vol_dict[pattern_str][1] += current_n_o_t
            vol_dict[pattern_str][2].append(hst_lst)
            vol_dict[pattern_str][3].append(current_close)
        else:
            vol_dict[pattern_str] = [
                vol, current_n_o_t, [hst_lst], [current_close]]
    for i in range(df_len):
        current_patt = df.iloc[i]['sign']
        vol += df['volume'].iat[i]
        n_o_t +=  df['num_of_trades'].iat[i]
        higest_list.append(df['high'].iat[i])
        lowest_list.append(df['low'].iat[i])
        if patt_temp == 1 and current_patt == -1:
            pattern[1] += 1

        elif patt_temp == -1 and current_patt == -1:
            pattern[1] += 1

        elif patt_temp == 1 and current_patt == 1:
            pattern[0] += 1

        elif patt_temp == -1 and current_patt == 1:
            current_close = df['close'].iat[i]
            add_to_pattern_dict(pattern, vol, n_o_t, max(higest_list)-min(lowest_list), current_close)

            pattern = [1, 0]
            vol = 0
            n_o_t = 0
            higest_list.clear()
            lowest_list.clear()

        patt_temp = current_patt

    patterns = []
    for patt in pattern_list:
        if patt in patterns:
            continue
        patterns.append(patt)
    for pattern in patterns:
        count = pattern_list.count(pattern)
        patt_from_dict = vol_dict[str(pattern[0]) + '-' + str(pattern[1])]
        vol_avg = patt_from_dict[0] / (count*sum(pattern))
        n_o_t_avg = patt_from_dict[1] / (count*sum(pattern))
        highest_lowest_avg = sum(patt_from_dict[2]) / len(patt_from_dict[2])
        avg_price = sum(patt_from_dict[3]) / len(patt_from_dict[3])
        price_vol_not = (avg_price * vol_avg) / n_o_t_avg if n_o_t_avg else 0
        pattern.extend(
            [vol_avg, n_o_t_avg, highest_lowest_avg, price_vol_not, count, patt_from_dict])
    pattern_df = pd.DataFrame(patterns, columns=[
                              '+', '-', 'vol_avg', 'n_o_t_avg', 'highest_lowest_avg', 'price*vol/not', 'count', 'pattern_str'])
    pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    pattern_df['percent'] = (pattern_df['count'] /
                             pattern_df['count'].sum()) * 100
    pattern_df['time'] = st_time
    pattern_df = pattern_df[['time', '+', '-', 'vol_avg', 'n_o_t_avg',
                             'highest_lowest_avg', 'price*vol/not', 'count', 'percent', 'pattern_str']]
    return pattern_df


def find_negative_patterns(df):
    print('Finding candle patterns')
    df_len = len(df)
    vol_dict = {}
    st_time = str(datetime.fromtimestamp(df['time'].iat[0]/1000))[:19]
    pattern_list = []
    pattern = [0, 0]
    patt_temp = -1
    vol = 0
    n_o_t = 0
    higest_list = []
    lowest_list = []


    def add_to_pattern_dict(pattern, vol, current_n_o_t, hst_lst, current_close):
        pattern_str = str(pattern[0]) + '-' + str(pattern[1])
        pattern_list.append(pattern)
        if pattern_str in vol_dict:
            vol_dict[pattern_str][0] += vol
            vol_dict[pattern_str][1] += current_n_o_t
            vol_dict[pattern_str][2].append(hst_lst)
            vol_dict[pattern_str][3].append(current_close)
        else:
            vol_dict[pattern_str] = [
                vol, current_n_o_t, [hst_lst], [current_close]]
    for i in range(df_len):
        current_patt = df.iloc[i]['sign']
        vol += df['volume'].iat[i]
        n_o_t +=  df['num_of_trades'].iat[i]
        higest_list.append(df['high'].iat[i])
        lowest_list.append(df['low'].iat[i])
        if patt_temp == -1 and current_patt == 1:
            pattern[1] += 1

        elif patt_temp == 1 and current_patt == 1:
            pattern[1] += 1

        elif patt_temp == -1 and current_patt == -1:
            pattern[0] += 1

        elif patt_temp == 1 and current_patt == -1:
            current_close = df['close'].iat[i]
            add_to_pattern_dict(pattern, vol, n_o_t, max(higest_list)-min(lowest_list), current_close)

            pattern = [1, 0]
            vol = 0
            n_o_t = 0
            higest_list.clear()
            lowest_list.clear()

        patt_temp = current_patt

    patterns = []
    for patt in pattern_list:
        if patt in patterns:
            continue
        patterns.append(patt)
    for pattern in patterns:
        count = pattern_list.count(pattern)
        patt_from_dict = vol_dict[str(pattern[0]) + '-' + str(pattern[1])]
        vol_avg = patt_from_dict[0] / (count*sum(pattern))
        n_o_t_avg = patt_from_dict[1] / (count*sum(pattern))
        highest_lowest_avg = sum(patt_from_dict[2]) / len(patt_from_dict[2])
        avg_price = sum(patt_from_dict[3]) / len(patt_from_dict[3])
        price_vol_not = (avg_price * vol_avg) / n_o_t_avg if n_o_t_avg else 0
        pattern.extend(
            [vol_avg, n_o_t_avg, highest_lowest_avg, price_vol_not, count, patt_from_dict])
    pattern_df = pd.DataFrame(patterns, columns=[
                              '-', '+', 'vol_avg', 'n_o_t_avg', 'highest_lowest_avg', 'price*vol/not', 'count', 'pattern_str'])
    pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    pattern_df['percent'] = (pattern_df['count'] /
                             pattern_df['count'].sum()) * 100
    pattern_df['time'] = st_time
    pattern_df = pattern_df[['time', '-', '+', 'vol_avg', 'n_o_t_avg',
                             'highest_lowest_avg', 'price*vol/not', 'count', 'percent', 'pattern_str']]
    return pattern_df


def calculate_chunked_patterns(df, open_source, close_source, num_of_chunks):
    positive_patterns_all = find_positive_patterns(
        df, open_source, close_source)
    negative_patterns_all = find_negative_patterns(
        df, open_source, close_source)
    negative_patterns_temp = pd.DataFrame()
    len_of_chunk = len(df) // num_of_chunks + 1
    positive_patterns_chunks = []
    negative_patterns_chunks = []
    for i in range(num_of_chunks):
        temp_df = df[i*len_of_chunk:(i+1)*len_of_chunk]
        positive_patt = find_positive_patterns(
            temp_df, open_source, close_source)
        negative_patt = find_negative_patterns(
            temp_df, open_source, close_source)
        positive_patterns_chunks.append(positive_patt)
        negative_patterns_chunks.append(negative_patt)
    positive_patterns_df = pd.DataFrame()
    negative_patterns_df = pd.DataFrame()
    for i in range(len(positive_patterns_all)):
        patt = positive_patterns_all.iloc[i]
        # _______________________________________________________________
        count_all = patt['count']
        patt['percent'] = 100
        positive_patterns_df = positive_patterns_df.append(patt)
        for patterns in positive_patterns_chunks:
            chunk_patt = patterns[patterns['+'] ==
                                  patt['+']][patterns['-'] == patt['-']]
            if len(chunk_patt) > 0:
                chunk_patt['percent'] = (chunk_patt['count'] / count_all) * 100
                positive_patterns_df = positive_patterns_df.append(chunk_patt)
        # _______________________________________________________________
        neg_pattern = negative_patterns_all[negative_patterns_all['+']
                                            == patt['+']][negative_patterns_all['-'] == patt['-']]
        if len(neg_pattern) > 0:
            negative_patterns_temp = negative_patterns_temp.append(neg_pattern)
            negative_patterns_all.drop(neg_pattern.index, inplace=True)
    negative_patterns_all = pd.concat(
        [negative_patterns_temp, negative_patterns_all])

    for i in range(len(negative_patterns_all)):
        patt = negative_patterns_all.iloc[i]
        count_all = patt['count']
        patt['percent'] = 100
        negative_patterns_df = negative_patterns_df.append(patt)
        for patterns in positive_patterns_chunks:
            chunk_patt = patterns[patterns['+'] ==
                                  patt['+']][patterns['-'] == patt['-']]
            if len(chunk_patt) > 0:
                chunk_patt['percent'] = (chunk_patt['count'] / count_all) * 100
                negative_patterns_df = negative_patterns_df.append(chunk_patt)
    del positive_patterns_all['time']
    del negative_patterns_all['time']
    return positive_patterns_all, negative_patterns_all, positive_patterns_df, negative_patterns_df


def plot_positive_tree_graph(df, depth=8, fname='positive_patterns_tree.png'):
    # max pattern num: 2 * max(max(df['+']), max(df['-']))
    number_of_nodes = (2**depth) - 1
    g = Graph.Tree(number_of_nodes, 2)
    layout = g.layout("tree")
    percent_list = [100, ]
    graph_node_color_list = ['green']
    patterns_str_list = ['+']
    temp_patterns_str_list = []
    for i in range(1, depth):
        for patt in patterns_str_list:
            graph_node_color_list += ['green', 'red']
            pos_patt = patt + '+'
            neg_patt = patt + '-'
            if pos_patt.find('-+') == -1:
                pos_patt_row = df[df['+'] >=
                                  pos_patt.count('+')][df['-'] >= pos_patt.count('-')]
                if len(pos_patt_row) > 0:
                    pos_patt_percent = pos_patt_row['percent'].iloc[0]
                else:
                    pos_patt_percent = 0
            else:
                pos_patt_percent = 0
            if neg_patt.find('-+') == -1:
                neg_patt_row = df[df['+'] ==
                                  neg_patt.count('+')][df['-'] >= neg_patt.count('-')]
                if len(neg_patt_row) > 0:
                    neg_patt_percent = neg_patt_row['percent'].iloc[0]
                else:
                    neg_patt_percent = 0
            else:
                neg_patt_percent = 0
            percent_sum = pos_patt_percent + neg_patt_percent
            if percent_sum != 0:
                percent_list += [f'{(pos_patt_percent*100/percent_sum):.2f}%',
                                 f'{(neg_patt_percent*100/percent_sum):.2f}%']
            else:
                percent_list += ['0', '0']

            temp_patterns_str_list += [pos_patt, neg_patt]
        patterns_str_list = temp_patterns_str_list
        temp_patterns_str_list = []

    plot(g, layout=layout,
         bbox=(5000, 5000),
         vertex_color=graph_node_color_list,
         vertex_label=percent_list,
         vertex_size=70,
         target=fname)


def plot_negative_tree_graph(df, depth=8, fname='negative_patterns_tree.png'):
    # max pattern num: 2 * max(max(df['+']), max(df['-']))
    number_of_nodes = (2**depth) - 1
    g = Graph.Tree(number_of_nodes, 2)
    layout = g.layout("tree")
    percent_list = [100, ]
    graph_node_color_list = ['red']
    patterns_str_list = ['-']
    temp_patterns_str_list = []
    for i in range(1, depth):
        for patt in patterns_str_list:
            graph_node_color_list += ['red', 'green']
            neg_patt = patt + '-'
            pos_patt = patt + '+'
            if neg_patt.find('+-') == -1:
                neg_patt_row = df[df['-'] ==
                                  neg_patt.count('-')][df['+'] >= neg_patt.count('+')]
                if len(neg_patt_row) > 0:
                    neg_patt_percent = neg_patt_row['percent'].iloc[0]
                else:
                    neg_patt_percent = 0
            else:
                neg_patt_percent = 0
            if pos_patt.find('+-') == -1:
                pos_patt_row = df[df['-'] >=
                                  pos_patt.count('-')][df['+'] >= pos_patt.count('+')]
                if len(pos_patt_row) > 0:
                    pos_patt_percent = pos_patt_row['percent'].iloc[0]
                else:
                    pos_patt_percent = 0
            else:
                pos_patt_percent = 0
            percent_sum = pos_patt_percent + neg_patt_percent
            if percent_sum != 0:
                percent_list += [f'{(neg_patt_percent*100/percent_sum):.2f}%',
                                 f'{(pos_patt_percent*100/percent_sum):.2f}%']
            else:
                percent_list += ['0', '0']

            temp_patterns_str_list += [pos_patt, neg_patt]
        patterns_str_list = temp_patterns_str_list
        temp_patterns_str_list = []

    plot(g, layout=layout,
         bbox=(5000, 5000),
         vertex_color=graph_node_color_list,
         vertex_label=percent_list,
         vertex_size=50,
         target=fname)





