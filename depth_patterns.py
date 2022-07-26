from igraph import *
import pandas as pd
import copy


"""
Find candle sign patterns with depth:
    example:
    we have candles with this pattern:
        ++-+--
        (+ indicates green and - indicates red candle)
        patterns will be:
        ++,
        ++-,
        +-,
        +--,
        +-
    at the end we count patterns and aggregate their statistic parameters
    the plot them and make a dataframe result
"""

def find_patterns_with_depth_(df: pd.DataFrame, depth):
    print(f'find patterns with depth {depth}')
    if depth > len(df):
        depth = len(df)

    df['sign'] = df.apply(lambda row: '+' if row.sign == 1 else '-', axis=1)
    patterns = {}
    for i in range(depth, len(df)):
        rows_df = df[i-depth:i]
        for j in range(1, depth):
            temp_df = rows_df.tail(j+1)
            pattern = ''.join(temp_df['sign'].tolist())
            hst_lst = (temp_df['high'].max() - temp_df['low'].min()) / temp_df['low'].min()
            vol = temp_df['volume'].sum()
            n_o_t = temp_df['num_of_trades'].sum()
            close = temp_df['close'].iloc[-1]
            if pattern not in patterns:
                patterns[pattern] = [1, vol, n_o_t, hst_lst, close]
            else:
                patterns[pattern][0] += 1
                patterns[pattern][1] += vol
                patterns[pattern][2] += n_o_t
                patterns[pattern][3] += hst_lst
                patterns[pattern][4] += close
    patts = []
    for patt, vals in patterns.items():
        count = vals[0]
        vol_avg = vals[1]/count
        n_o_t_avg = vals[2]/count
        hst_lst_avg = vals[3]/count
        avg_price = vals[4]/count
        price_vol_not = (avg_price * vol_avg) / n_o_t_avg if n_o_t_avg else 0
        patts.append([patt, count, vol_avg, n_o_t_avg, hst_lst_avg, price_vol_not])
    patterns_df = pd.DataFrame(patts, columns=['pattern', 'count', 'vol_avg', 'n_o_t_avg', 'highest_lowest_avg%', 'price*vol/not'])
    # patterns_df['percent'] = patterns_df['count']/patterns_df['count'].sum() * 100
    # patterns_df = patterns_df.sort_values(by=['count'], ascending=False)
    return patterns_df


def find_patterns_with_depth(df: pd.DataFrame, depth):
    result = find_patterns_with_depth_(df, depth)
    result['percent'] = result['count']/result['count'].sum() * 100
    result = result.sort_values(by=['count'], ascending=False)
    return result

def unique_patterns(df: pd.DataFrame):
    result = df.drop_duplicates(subset='pattern')
    result['count'] = result.apply(lambda row: df[df.pattern==row.pattern]['count'].sum(), axis=1)
    result['percent'] = result['count']/result['count'].sum() * 100
    result = result.sort_values(by=['count'], ascending=False)
    return result



def plot_depth_pattern_graph(df, depth=7, fname='depth_patterns_graph'):
    number_of_nodes = (2**depth) - 1
    pos_g = Graph.Tree(number_of_nodes, 2)
    neg_g = Graph.Tree(number_of_nodes, 2)
    layout_1 = pos_g.layout("tree")
    layout_2 = neg_g.layout("tree")
    pos_percent_list = [100, ]
    neg_percent_list = [100, ]
    pos_node_color_list = ['green']
    neg_node_color_list = ['red']
    pos_patterns_str_list = ['+']
    neg_patterns_str_list = ['-']
    pos_temp_patterns_str_list = []
    neg_temp_patterns_str_list = []
    for i in range(1, depth):
        for patt in pos_patterns_str_list:
            pos_node_color_list += ['green', 'red']
            pos_patt = patt + '+'
            neg_patt = patt + '-'
            pos_patt_row = df[df['pattern'].str.find(pos_patt) == 0]
            pos_patt_percent = pos_patt_row['percent'].sum() if len(pos_patt_row) > 0 else 0
            neg_patt_row = df[df['pattern'].str.find(neg_patt) == 0]
            neg_patt_percent = neg_patt_row['percent'].sum() if len(neg_patt_row) > 0 else 0

            percent_sum = pos_patt_percent + neg_patt_percent
            if percent_sum != 0:
                pos_percent_list += [f'{(pos_patt_percent*100/percent_sum):.2f}%',
                                 f'{(neg_patt_percent*100/percent_sum):.2f}%']
            else:
                pos_percent_list += ['0', '0']

            pos_temp_patterns_str_list += [pos_patt, neg_patt]

        for patt in neg_patterns_str_list:
            neg_node_color_list += ['red', 'green']
            neg_patt = patt + '-'
            pos_patt = patt + '+'
            neg_patt_row = df[df['pattern'].str.find(neg_patt) == 0]
            neg_patt_percent = neg_patt_row['percent'].sum() if len(neg_patt_row) > 0 else 0
            pos_patt_row = df[df['pattern'].str.find(pos_patt) == 0]
            pos_patt_percent = pos_patt_row['percent'].sum() if len(pos_patt_row) > 0 else 0

            percent_sum = pos_patt_percent + neg_patt_percent
            if percent_sum != 0:
                neg_percent_list += [f'{(neg_patt_percent*100/percent_sum):.2f}%',
                                 f'{(pos_patt_percent*100/percent_sum):.2f}%']
            else:
                neg_percent_list += ['0', '0']

            neg_temp_patterns_str_list += [neg_patt, pos_patt]

        pos_patterns_str_list = pos_temp_patterns_str_list
        neg_patterns_str_list = neg_temp_patterns_str_list
        pos_temp_patterns_str_list = []
        neg_temp_patterns_str_list = []
    plot(pos_g, layout=layout_1,
         bbox=(5000, 5000),
         vertex_color=pos_node_color_list,
         vertex_label=pos_percent_list,
         vertex_size=70,
         target=fname+'_positive.png')
    plot(neg_g, layout=layout_2,
         bbox=(5000, 5000),
         vertex_color=neg_node_color_list,
         vertex_label=neg_percent_list,  
         vertex_size=70,
         target=fname+'_negative.png')



def plot_paired_depth_pattern_graph(df, depth=7, fname='paired_depth_patterns'):
    print(f'plotting paired depth patterns graph with depth {depth} with name {fname}')
    # max pattern num: 2 * max(max(df['+']), max(df['-']))
    number_of_nodes = 0
    for i in range(depth):
        number_of_nodes += 4**i
    number_of_nodes = int(number_of_nodes)
    pos_pos_g = Graph.Tree(number_of_nodes, 4)
    pos_neg_g = Graph.Tree(number_of_nodes, 4)
    neg_pos_g = Graph.Tree(number_of_nodes, 4)
    neg_neg_g = Graph.Tree(number_of_nodes, 4)
    layout_1 = pos_pos_g.layout("tree")
    layout_2 = pos_neg_g.layout("tree")
    layout_3 = neg_pos_g.layout("tree")
    layout_4 = neg_neg_g.layout("tree")
    # percent lists
    pos_pos_percent = df[df['pattern'].str.find('++') == 0]['percent'].sum()
    pos_neg_percent = df[df['pattern'].str.find('+-') == 0]['percent'].sum()
    neg_pos_percent = df[df['pattern'].str.find('-+') == 0]['percent'].sum()
    neg_neg_percent = df[df['pattern'].str.find('--') == 0]['percent'].sum()
    percent_sum = pos_pos_percent + pos_neg_percent + neg_pos_percent + neg_neg_percent
    pos_pos_percent_list = [f'++\n{(pos_pos_percent*100/percent_sum):.2f}%', ]
    pos_neg_percent_list = [f'+-\n{(pos_neg_percent*100/percent_sum):.2f}%', ]
    neg_pos_percent_list = [f'-+\n{(neg_pos_percent*100/percent_sum):.2f}%', ]
    neg_neg_percent_list = [f'--\n{(neg_neg_percent*100/percent_sum):.2f}%', ]
    # node color lists
    pos_pos_node_color_list = ['green']
    pos_neg_node_color_list = ['blue']
    neg_pos_node_color_list = ['yellow']
    neg_neg_node_color_list = ['red']
    # pattern as str lists
    pos_pos_patterns_str_list = ['++']
    pos_neg_patterns_str_list = ['+-']
    neg_pos_patterns_str_list = ['-+']
    neg_neg_patterns_str_list = ['--']
    # temp pattern as str lists
    pos_pos_temp_patterns_str_list = []
    pos_neg_temp_patterns_str_list = []
    neg_pos_temp_patterns_str_list = []
    neg_neg_temp_patterns_str_list = []

    def calculate_subpattern_results(patt, patt_percent_list, node_color_list, temp_patterns_str_list):
        node_color_list += ['green', 'blue', 'yellow', 'red']
        pos_pos_patt = patt + '++'
        pos_neg_patt = patt + '+-'
        neg_pos_patt = patt + '-+'
        neg_neg_patt = patt + '--'
        temp_patterns_str_list += [pos_pos_patt, pos_neg_patt, neg_pos_patt, neg_neg_patt]
        pos_pos_patt_row = df[df['pattern'].str.find(pos_pos_patt) == 0]
        pos_pos_patt_percent = pos_pos_patt_row['percent'].sum() if len(pos_pos_patt_row) > 0 else 0
        pos_neg_patt_row = df[df['pattern'].str.find(pos_neg_patt) == 0]
        pos_neg_patt_percent = pos_neg_patt_row['percent'].sum() if len(pos_neg_patt_row) > 0 else 0
        neg_pos_patt_row = df[df['pattern'].str.find(neg_pos_patt) == 0]
        neg_pos_patt_percent = neg_pos_patt_row['percent'].sum() if len(neg_pos_patt_row) > 0 else 0
        neg_neg_patt_row = df[df['pattern'].str.find(neg_neg_patt) == 0]
        neg_neg_patt_percent = neg_neg_patt_row['percent'].sum() if len(neg_neg_patt_row) > 0 else 0
        percent_sum = pos_pos_patt_percent + pos_neg_patt_percent + neg_pos_patt_percent + neg_neg_patt_percent
        if percent_sum != 0:
            patt_percent_list += [f'++\n{(pos_pos_patt_percent*100/percent_sum):.2f}%',
                                 f'+-\n{(pos_neg_patt_percent*100/percent_sum):.2f}%',
                                 f'-+\n{(neg_pos_patt_percent*100/percent_sum):.2f}%',
                                 f'--\n{(neg_neg_patt_percent*100/percent_sum):.2f}%']
        else:
            patt_percent_list += ['0', '0', '0', '0']



    for i in range(1, depth):
        for patt in pos_pos_patterns_str_list:
            calculate_subpattern_results(
                            patt, pos_pos_percent_list, pos_pos_node_color_list, pos_pos_temp_patterns_str_list)
        for patt in pos_neg_patterns_str_list:
            calculate_subpattern_results(
                            patt, pos_neg_percent_list, pos_neg_node_color_list, pos_neg_temp_patterns_str_list)
        for patt in neg_pos_patterns_str_list:
            calculate_subpattern_results(
                            patt, neg_pos_percent_list, neg_pos_node_color_list, neg_pos_temp_patterns_str_list)
        for patt in neg_neg_patterns_str_list:
            calculate_subpattern_results(
                            patt, neg_neg_percent_list, neg_neg_node_color_list, neg_neg_temp_patterns_str_list)

        pos_pos_patterns_str_list = copy.deepcopy(pos_pos_temp_patterns_str_list)
        pos_neg_patterns_str_list = copy.deepcopy(pos_neg_temp_patterns_str_list)
        neg_pos_patterns_str_list = copy.deepcopy(neg_pos_temp_patterns_str_list)
        neg_neg_patterns_str_list = copy.deepcopy(neg_neg_temp_patterns_str_list)

    plot(pos_pos_g, layout=layout_1,
         bbox=(7500, 7500),
         vertex_color=pos_pos_node_color_list,
         vertex_label=pos_pos_percent_list,
         vertex_size=70,
         target=fname+'++.png')
    plot(pos_neg_g, layout=layout_2,
            bbox=(7500, 7500),
            vertex_color=pos_neg_node_color_list,
            vertex_label=pos_neg_percent_list,
            vertex_size=70,
            target=fname+'+-.png')
    plot(neg_pos_g, layout=layout_3,
            bbox=(7500, 7500),
            vertex_color=neg_pos_node_color_list,
            vertex_label=neg_pos_percent_list,
            vertex_size=70,
            target=fname+'-+.png')
    plot(neg_neg_g, layout=layout_4,
            bbox=(7500, 7500),
            vertex_color=neg_neg_node_color_list,
            vertex_label=neg_neg_percent_list,
            vertex_size=70,
            target=fname+'--.png')