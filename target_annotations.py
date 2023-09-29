import streamlit as st
from utils import get_dataset, get_args, get_splits, get_environment, \
    get_instance_metadata, df_from_string, process_reports
import pandas as pd
import os
import pickle as pkl


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
    filtered_instance_metadata = filtered_instance_metadata[
        filtered_instance_metadata['target diagnosis countdown'].apply(
            lambda x: x == x and len(x[0]) > 0)]
    if st.checkbox('Only show instances with cached evidence', value=True):
        filtered_instance_metadata = filtered_instance_metadata[
            filtered_instance_metadata['is valid timestep'].apply(
                lambda x: x == x and sum(x) > 0)]
    if filter_instances_string != '':
        filtered_instance_metadata = filtered_instance_metadata[
            filtered_instance_metadata.apply(
                eval(filter_instances_string), axis=1)]
    valid_instances = filtered_instance_metadata.episode_idx
    num_valid = len(valid_instances)
    show_instances = st.checkbox('Show instances')
if show_instances:
    instances_expander = st.expander('instances')
    instances_expander.write("")
metadata_index, instance_index = st.selectbox(
    f'Instances ({num_valid})', list(enumerate(valid_instances)),
    format_func=lambda x: ' '.join(df.iloc[x[1]].instance_name.split()[:2]))
annotator_name = st.text_input('Annotator Name')
if annotator_name == '':
    st.warning(
        'You need to specify an annotator name to submit annotations.')
    st.stop()
target_diagnoses = []
countdowns = []
for target_diagnosis, countdown in filtered_instance_metadata.iloc[
        metadata_index]['target diagnosis countdown'][0].items():
    target_diagnoses.append(target_diagnosis)
    countdowns.append(countdown)
tabs = st.tabs(target_diagnoses)
annotations = {
    'instance_idx': instance_index, 'diagnosis_anns': {}, 'split': split}
for target_diagnosis, countdown, tab in zip(
        target_diagnoses, countdowns, tabs):
    annotations['diagnosis_anns'][target_diagnosis] = {}
    with tab:
        reports = df_from_string(df.iloc[instance_index].reports)
        reports['date'] = pd.to_datetime(reports['date'])
        reports = process_reports(reports)
        st.write('### ' + reports.iloc[countdown].report_name)
        st.write(f'Description: {reports.iloc[countdown].description}')
        st.divider()
        text = reports.iloc[countdown].text
        st.write(text.strip().replace('\n', '\\\n'))
        # st.text(text)
        st.write(f'### Extracted Diagnosis Annotation: {target_diagnosis}')
        is_confident_diagnosis_annotation = st.radio(
            f'Is \"{target_diagnosis}\" a confident diagnosis of the patient '
            'according to the report?', ['Yes', 'No'],
            key=f'q1 {instance_index} {target_diagnosis}')
        annotations['diagnosis_anns'][target_diagnosis][
            'report_idx'] = countdown
        annotations['diagnosis_anns'][target_diagnosis][
            'is_confident_diagnosis'] = is_confident_diagnosis_annotation
        if is_confident_diagnosis_annotation == 'Yes':
            could_be_identified_earlier = st.radio(
                'Is it likely that this confident diagnosis could be '
                'identified in earlier reports?', ['Yes', 'No'],
                key=f'q2 {instance_index} {target_diagnosis}')
            annotations['diagnosis_anns'][target_diagnosis][
                'could_be_identified_earlier'] = could_be_identified_earlier
with tab:
    submit_anns = st.button('Submit Annotations', key=f'submit {instance_index}')
if submit_anns:
    st.success(annotations)
    if not os.path.exists(args['annotations']['target_anns_path']):
        os.mkdir(args['annotations']['target_anns_path'])
    if not os.path.exists(os.path.join(
            args['annotations']['target_anns_path'], split)):
        os.mkdir(os.path.join(args['annotations']['target_anns_path'], split))
    ann_path = os.path.join(
        args['annotations']['target_anns_path'], split,
        '_'.join(annotator_name.split()))
    if not os.path.exists(ann_path):
        os.mkdir(ann_path)
    next_idx = 0
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            next_idx = max(
                next_idx, int(filename.split('.')[0].split('_')[1]) + 1)
    new_filepath = os.path.join(ann_path, f'ann_{next_idx}.pkl')
    st.success(f'Writing to: {new_filepath}')
    with open(new_filepath, 'wb') as f:
        pkl.dump({next_idx: annotations}, f)
with st.expander('Annotations'):
    ann_path = os.path.join(
        args['annotations']['target_anns_path'], split,
        '_'.join(annotator_name.split()))
    if not os.path.exists(ann_path):
        anns_df = pd.DataFrame([])
    else:
        anns_df = {}
        for filename in os.listdir(ann_path):
            if filename.startswith('ann_') and filename.endswith('.pkl'):
                with open(os.path.join(ann_path, filename), 'rb') as f:
                    anns_df.update(pkl.load(f))
        anns_df = pd.DataFrame(anns_df).transpose()
    st.write(anns_df)
if show_instances:
    with st.spinner('Writing table...'):
        instances_expander.write(filtered_instance_metadata)
