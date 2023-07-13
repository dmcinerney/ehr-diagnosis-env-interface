from utils import get_args
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import streamlit as st
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr


@st.cache_data
def get_cache(cache_dir):
    instances = []
    files = os.listdir(cache_dir)
    for file in files:
        with open(os.path.join(cache_dir, file), 'rb') as f:
            instance = pkl.load(f)
        key = next(iter(instance.keys()))
        instance = instance[key]
        instances.append(
            {'instance': key, 'length': len(next(iter(instance.values())))})
        instances[-1].update({
            k: set().union(
                *v) if isinstance(v[0], set) else set(v).difference({None})
            for k, v in instance.items() if len(v) > 0})
    return pd.DataFrame(instances)


def get_counts(cache_info):
    counters = {c: Counter()
                for c in cache_info.columns if c not in ['instance', 'length']}
    for i, row in cache_info.iterrows():
        for name, counter in counters.items():
            for x in row[name]:
                counter[x] += 1
    return counters


def plot_counts(cache_info, counters, counters_before_filtering=None):
    figures = {}
    for name, counter in counters.items():
        if name == 'total':
            continue
        print(name)
        string_frequency = [[k, v] for k, v in counter.items()]
        if counters_before_filtering is not None:
            for i, (k, tp) in enumerate(string_frequency):
                positives = counters_before_filtering[name][k]
                string_frequency[i].append(positives)
                fp = positives - tp
                fn = counters['total'] - tp
                tn = counters_before_filtering['total'] - positives - fn
                x = [1] * (tp + fp) + [0] * (fn + tn)
                y = [1] * tp + [0] * fp + [1] * fn + [0] * tn
                corr = pearsonr(x, y)
                string_frequency[i].append(corr)
            string_frequency = sorted(
                string_frequency, key=lambda x: -x[3].statistic)
        else:
            string_frequency = sorted(string_frequency, key=lambda x: -x[1])
        string_frequency = string_frequency[:20]
        print(string_frequency)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        strings = [
            (x[0] if len(x[0]) < 15 else x[0][:12] + '...')
            for x in string_frequency]
        if counters_before_filtering is not None:
            ax.barh(-np.arange(len(string_frequency)),
                    [x[2]*100/len(cache_info) for x in string_frequency])
            strings = [
                s + ' ({:.2f}±{:.2f})'.format(x[3].statistic, x[3].pvalue)
                for s, x in zip(strings, string_frequency)]
        ax.barh(-np.arange(len(string_frequency)),
                [x[1]*100/len(cache_info) for x in string_frequency])
        ax.set_yticks(-np.arange(len(string_frequency)), strings)
        ax.set_xlabel('Counts')
        ax.set_title(f'{name} counts')
        figures[name] = fig
    return figures


@st.cache_data
def get_and_plot_counts(_cache_info, filter=None):
    cache_info_filtered = _cache_info
    if filter is not None:
        filter = eval(filter)
        cache_info_filtered = cache_info_filtered[cache_info_filtered.apply(
            filter, axis=1)]
    st.write(f'Total Dataset Size: {len(_cache_info)}')
    st.write('Size after filtering: {} ({:.0f}%)'.format(
        len(cache_info_filtered),
        len(cache_info_filtered) * 100 / len(_cache_info)))
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    sns.histplot(data=cache_info_filtered, x='length')
    st.pyplot(fig)
    counters = get_counts(cache_info_filtered)
    counters['total'] = len(cache_info_filtered)
    counters_before_filtering = None
    if filter is not None:
        counters_before_filtering = get_counts(_cache_info)
        counters_before_filtering['total'] = len(_cache_info)
    figures = plot_counts(_cache_info, counters,
                          counters_before_filtering=counters_before_filtering)
    num_columns = 3
    columns = st.columns(num_columns)
    for i, (k, fig) in enumerate(figures.items()):
        with columns[i % num_columns]:
            st.write(f'Top {k}s')
            st.pyplot(fig)


@st.cache_resource
def get_fuzzy_matching_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_data
def get_fuzzy_matches(term, options, threshold=.85):
    model = get_fuzzy_matching_model()
    embeddings = model.encode([term] + options)
    cosine_sim = util.cos_sim(embeddings[:1], embeddings[1:])
    return [option for option, sim in zip(options, cosine_sim.squeeze(0))
            if sim > threshold]


st.set_page_config(layout="wide")
st.title('Dataset Overview')
args = get_args('config.yaml')
split = st.selectbox('Select the split', ['train'])
# split = st.selectbox('Select the split', ['train', 'val', 'test'])
cache_info = get_cache(args['data'][f'{split}_cache_path'])
columns_to_filter = {}
counters = get_counts(cache_info)
set_columns = [c for c in cache_info.columns if c not in [
    'instance', 'length']]
add_fuzzy_matching_terms = st.checkbox('Add fuzzy matching terms')
for column in set_columns:
    options = sorted(list(set().union(
        *cache_info[column].tolist())), key=lambda x: -counters[column][x])
    terms = st.multiselect(
        f'Filter to patients that include the selected word in the '
        f'\"{column}\"',
        options)
    if len(terms) > 0:
        if add_fuzzy_matching_terms:
            options.remove(terms)
            fuzzy_matches = get_fuzzy_matches(terms, options)
            terms += fuzzy_matches
        st.write('Terms: ' + ', '.join(terms))
        columns_to_filter[column] = terms
filter = None if len(columns_to_filter) == 0 else \
    'lambda r: all([any([term in r[k] for term in v]) ' \
    'for k, v in {}.items()])'.format(
        str(columns_to_filter))
print(filter)
get_and_plot_counts(cache_info, filter=filter)
