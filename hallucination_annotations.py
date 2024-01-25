from utils import get_full_evidence_df
import streamlit as st
from utils import get_dataset, get_args, df_from_string, \
    get_all_hallucination_annotations
import pandas as pd
import os
import pickle as pkl


def normalize_string(string):
    return ' '.join(string.strip().lower().split())


@st.cache_resource
def get_instances(_dataset_split, _annotated_evidence, split):
    instances = []
    seen_identifiers = set()
    for i, row in _annotated_evidence.iterrows():
        if row.evidence_anns == row.evidence_anns:
            reports = df_from_string(_dataset_split[
                _dataset_split.instance_name == row.instance[len(split) + 1:]
                ].reports.iloc[0])
            for idx, evidence in row.evidence_anns.items():
                if row['model_type'] != 'llm_evidence':
                    continue
                report = dict(
                    reports.iloc[
                        int(evidence['evidence']['report_number']) - 1])
                normalized_report_text = normalize_string(report['text'])
                if normalize_string(evidence['evidence']['evidence']) in \
                        normalized_report_text:
                    continue
                if '\nSigns:' in evidence['evidence']['evidence']:
                    ev1, ev2 = evidence['evidence']['evidence'].split(
                        '\nSigns:')
                    ev1, ev2 = normalize_string(ev1), normalize_string(ev2)
                    if ev1 in normalized_report_text and \
                            ev2 in normalized_report_text:
                        continue
                identifier = row['instance'] + \
                    ' ModelType ' + row['model_type'].replace(' ', '_') + \
                    ' SortType ' + row['sort_type'].replace(' ', '_') + \
                    ' EvidenceIdx ' + str(idx)
                if identifier in seen_identifiers:
                    continue
                seen_identifiers.add(identifier)
                instances.append({
                    'identifier': identifier,
                    'instance': row['instance'],
                    'model_type': row['model_type'],
                    'sort_type': row['sort_type'],
                    'evidence_idx': idx,
                    'evidence': evidence['evidence']['evidence'],
                    'evidence_was_seen': evidence['evidence_was_seen']
                        if 'evidence_was_seen' in evidence.keys() else None,
                    'report_number': evidence['evidence']['report_number'],
                    'report': report,
                })
    return pd.DataFrame(instances)


st.set_page_config(layout="wide")
st.title('Hallucination Annotations')
args = get_args('config.yaml')
annotated_evidence, annotator_instance_repeats = get_full_evidence_df(
    args['annotations']['hallucination_evidence_paths'])
assert len(annotator_instance_repeats) == 0
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = list(annotated_evidence.keys())
with st.sidebar:
    split = st.selectbox(
        'Dataset Split', splits,
        index=splits.index('val1') if 'val1' in splits else 0)
    dataset_split = get_dataset(args, split)
    df = get_instances(dataset_split, annotated_evidence[split], split)
    valid_instances = df.identifier
    num_valid = len(valid_instances)
anns_df = get_all_hallucination_annotations(args, split)
annotator_name = st.text_input('Annotator Name')
def format_func(x):
    identifier_parts = x[1].split()
    name = ' '.join(identifier_parts[1:3] + identifier_parts[-2:])
    if 'identifier' in anns_df.columns and \
            x[1] in set(anns_df.identifier):
        annotators = set(anns_df[anns_df.identifier == x[1]].annotator)
        name += ' ({})'.format(', '.join([
            'You' if '_'.join(annotator_name.split()) == ann else ann
            for ann in annotators]))
    return name
default_index = 0 if 'last_instance' not in st.session_state.keys() else \
    list(valid_instances).index(st.session_state['last_instance'])
index, identifier = st.selectbox(
    f'Instances ({num_valid})',
    list(enumerate(valid_instances)),
    index=default_index,
    format_func=format_func) # type: ignore
if annotator_name == '':
    st.warning(
        'You need to specify an annotator name to submit annotations.')
    st.stop()
row = df.iloc[index]
annotations = {
    **dict(row),
    'split': split}
st.write('### Evidence')
st.write('##### ' + row.evidence.replace('\n', '\n##### '))
st.write('### Hallucination Annotation')
annotations['hallucination_ann'] = st.radio(
    f'Is the above piece of evidence hallucinated? '
    '(You can reference the report it came from below.)',
    ['No', 'Partially', 'Yes'])
st.write('### ' + row.report['report_type'])
description = row.report['description']
st.write(f'Description: {description}')
st.divider()
text = row.report['text']
st.write(text.strip().replace('\n', '\\\n'))
# st.text(text)
def submit_annotations(args, split, annotator_name, annotations):
    if not os.path.exists(args['annotations']['hallucination_anns_path']):
        os.mkdir(args['annotations']['hallucination_anns_path'])
    if not os.path.exists(os.path.join(
            args['annotations']['hallucination_anns_path'], split)):
        os.mkdir(os.path.join(
            args['annotations']['hallucination_anns_path'], split))
    ann_path = os.path.join(
        args['annotations']['hallucination_anns_path'], split,
        '_'.join(annotator_name.split()))
    if not os.path.exists(ann_path):
        os.mkdir(ann_path)
    next_idx = 0
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            next_idx = max(
                next_idx, int(filename.split('.')[0].split('_')[1]) + 1)
    new_filepath = os.path.join(ann_path, f'ann_{next_idx}.pkl')
    # st.success(f'Writing to: {new_filepath}')
    print(f'Writing to: {new_filepath}')
    with open(new_filepath, 'wb') as f:
        pkl.dump({next_idx: annotations}, f)
class SubmitAnnotations:
    def __init__(self, args, split, annotator_name, annotations):
        self.args = args
        self.split = split
        self.annotator_name = annotator_name
        self.annotations = annotations
    def __call__(self):
        submit_annotations(
            self.args, self.split, self.annotator_name, self.annotations)
        st.session_state['last_instance'] = self.annotations['identifier']
submit_anns = st.button(
    'Submit Annotations', key=f'submit {identifier}',
    on_click=SubmitAnnotations(args, split, annotator_name, annotations))
if submit_anns:
    st.success('Submitted!')
with st.expander('Annotations'):
    anns_df = get_all_hallucination_annotations(args, split)
    st.write(anns_df)
