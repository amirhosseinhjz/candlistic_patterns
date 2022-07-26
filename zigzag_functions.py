import pandas as pd
from zigzag import *
import matplotlib.pyplot as plt

HH, LL, LH, HL = '1', '2', '3', '4'
patt_dict = {HH: 'HH', HL: 'HL', LL: 'LL', LH: 'LH'}
def calculate_zigzag(df, source, accuracy, interval):
    accuracy = abs(accuracy)
    src = np.array(df[source])
    pivots = peak_valley_pivots(src, accuracy, -accuracy)
    plot_pivots(src, pivots, f'{source}_zigzag_{interval}.png')
    pivots = pd.Series(pivots)
    df[f'{source}_zigzag'] = pivots
    return df

def find_zigzag_patterns(df, source, peak_valley_pattern):
    pattern_name = 'peak' if peak_valley_pattern == 1 else 'valley'
    print(f'Finding zigzag patterns for {source=}, {pattern_name}')
    peak_valleys = df[df[f'{source}_zigzag'] == peak_valley_pattern]
    # peak_valleys = peak_valleys[f'{source}_zigzag']
    pattern_list = []
    pattern = 0
    if peak_valley_pattern == 1:
        patt_temp = peak_valleys.iloc[0][source]
        for i in range(len(peak_valleys)):
            current_peak_valley = peak_valleys.iloc[i][source]
            if current_peak_valley >= patt_temp:
                pattern += 1
            else:
                pattern_list.append(pattern)
                pattern = 1
            patt_temp = current_peak_valley
        pattern_list.append(pattern)

    elif peak_valley_pattern == -1:
        patt_temp = peak_valleys.iloc[0][source]
        for i in range(len(peak_valleys)):
            current_peak_valley = peak_valleys.iloc[i][source]
            if current_peak_valley <= patt_temp:
                pattern += 1
            else:
                pattern_list.append(pattern)
                pattern = 1
            patt_temp = current_peak_valley
        pattern_list.append(pattern)

    patterns_set = set(pattern_list)
    patterns = []
    for pattern in patterns_set:
        patterns.append([pattern, pattern_list.count(pattern)])
    pattern_df = pd.DataFrame(patterns, columns=[pattern_name, 'count'])
    pattern_df['percent'] = (pattern_df['count'] / pattern_df['count'].sum()) * 100
    pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    return pattern_df

def calculate_mix_zigzag(df, accuracy, interval):
    df = calculate_zigzag(df, 'high', accuracy, interval)
    df = calculate_zigzag(df, 'low', accuracy, interval)
    hz = df[df['high_zigzag']!=0].iloc[1]
    lz = df[df['low_zigzag']!=0].iloc[1]
    zigzag_starter = hz if hz['time'] < lz['time'] else lz
    idx = df[df['time'] == zigzag_starter['time']].index.tolist()[0]
    last_exterm = zigzag_starter['high_zigzag'] if zigzag_starter['high_zigzag'] != 0 else zigzag_starter['low_zigzag']
    df_len = len(df)
    zigzag = np.array([0]*df_len)
    zigzag[idx] = last_exterm
    for i in range(idx, df_len):
        current_candle = df.iloc[i]
        high_zigzag = current_candle['high_zigzag']
        low_zigzag = current_candle['low_zigzag']
        if not any([high_zigzag, low_zigzag]):
            continue
        if high_zigzag != 0 and high_zigzag == -last_exterm:
            last_exterm = high_zigzag
            zigzag[i] = high_zigzag
        elif low_zigzag != 0 and low_zigzag == -last_exterm:
            last_exterm = low_zigzag
            zigzag[i] = low_zigzag
    plot_pivots(np.array(df['close']), zigzag, f'zigzag_mix_{interval}.png')
    df['zigzag'] = zigzag
    return df

def find_patterns_count_len_1(patterns: str):
    ls = []
    for i in [HH, LH, LL, HL]:
        ls.append([patt_dict[i], patterns.count(i)])
    pattern_df = pd.DataFrame(ls, columns=['pattern', 'count'])
    pattern_df['percent'] = (pattern_df['count'] / pattern_df['count'].sum()) * 100
    # pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    return pattern_df


def find_patterns_count_len_2(patterns):
    ls = []
    for i in [HH, LH, LL, HL]:
        sec_list = [HH,LH] if i in [LL, HL] else [LL, HL]
        for j in sec_list:
            ls.append([f'{patt_dict[i]}, {patt_dict[j]}', patterns.count(f'{i}{j}')])

    pattern_df = pd.DataFrame(ls, columns=['pattern', 'count'])
    pattern_df['percent'] = (pattern_df['count'] / pattern_df['count'].sum()) * 100
    # pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    return pattern_df

def find_patterns_count_len_3(patterns):
    ls = []
    percent_dict = {}
    for i in [HH, LH, LL, HL]:
        sec_list = [HH,LH] if i in [LL, HL] else [LL, HL]
        for j in sec_list:
            third_list = [HH, LH] if j in [LL, HL] else [LL, HL]
            percent_dict[f'{patt_dict[i]}, {patt_dict[j]}'] = {}
            for k in third_list:
                count = patterns.count(f'{i}{j}{k}')
                ls.append([f'{patt_dict[i]}, {patt_dict[j]}, {patt_dict[k]}', count])
                percent_dict[f'{patt_dict[i]}, {patt_dict[j]}'][patt_dict[k]] = count
            
    pattern_df = pd.DataFrame(ls, columns=['pattern', 'count'])
    pattern_df['percent'] = (pattern_df['count'] / pattern_df['count'].sum()) * 100
    # pattern_df = pattern_df.sort_values(by=['count'], ascending=False)
    return pattern_df

def calculate_zigzag_patterns(df, source):
    print(f'Finding zigzag patterns for {source=}')
    zigzags = df[df[source] != 0]
    patterns = ''
    hh = zigzags.iloc[0]['high']
    ll = zigzags.iloc[0]['low']
    for i in range(len(zigzags)):
        current_cand = zigzags.iloc[i]
        if current_cand[source] == 1:
            if current_cand['high'] > hh:
                patterns+= HH
            else:
                patterns+= LH
            hh = current_cand['high']
        elif current_cand[source] == -1:
            if current_cand['low'] < ll:
                patterns+= LL
            else:
                patterns+= HL
            ll = current_cand['low']
    # zigzags['patterns'] = patterns
    patts_1 = find_patterns_count_len_1(patterns)
    patts_2 = find_patterns_count_len_2(patterns)
    patts_3 = find_patterns_count_len_3(patterns)
    patts = pd.concat([patts_1, patts_2, patts_3])
    return patts


def plot_pivots(X, pivots, fname):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
    plt.savefig(fname=fname, dpi=300)
    plt.clf()