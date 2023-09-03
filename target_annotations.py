import streamlit as st
from utils import get_dataset, get_args, get_splits, get_environment, \
    get_instance_metadata, df_from_string


st.set_page_config(layout="wide")
st.title('Diagnosis Extraction Annotations')
args = get_args('config.yaml')
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = get_splits(args)
with st.sidebar:
    split = st.selectbox(
        'Dataset Split', splits,
        index=splits.index('val1') if 'val1' in splits else 0)
    df = get_dataset(args, split)
    env = get_environment(
        args, split, df, None, None, fuzzy_matching_threshold=None)
    instance_metadata = get_instance_metadata(env, args, split)
    filter_instances_string = st.text_input(
        'Type a lambda expression in python that filters instances using their'
        ' cached metadata.')
    filtered_instance_metadata = instance_metadata
    if st.checkbox('Only show instances with cached evidence', value=True):
        filtered_instance_metadata = filtered_instance_metadata[
            ~filtered_instance_metadata.day.isna()]
    if filter_instances_string != '':
        filtered_instance_metadata = filtered_instance_metadata[
            filtered_instance_metadata.apply(
                eval(filter_instances_string), axis=1)]
    valid_instances = filtered_instance_metadata.episode_idx
    num_valid = len(valid_instances)
instances_expander = st.expander('instances')
instances_expander.write("")
metadata_index, instance_index = st.selectbox(
    f'Instances ({num_valid})', list(enumerate(valid_instances)),
    format_func=lambda x: ' '.join(df.iloc[x[1]].instance_name.split()[:2]))
for target_diagnosis, countdown in filtered_instance_metadata.iloc[
        metadata_index]['target diagnosis countdown'][0].items():
    st.write('### Report')
    st.write(
        df_from_string(df.iloc[instance_index].reports).iloc[countdown].text)
    st.write(f'### Extracted Diagnosis Annotation: {target_diagnosis}')
with st.spinner('Writing table...'):
    instances_expander.write(filtered_instance_metadata)
