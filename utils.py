from omegaconf import OmegaConf
import pandas as pd
import os
import streamlit as st
import io
import ehr_diagnosis_env
from ehr_diagnosis_env.utils import get_model_interface
from sentence_transformers import SentenceTransformer
import gymnasium
from stqdm import stqdm
from collections import defaultdict
import pickle as pkl
import os


# Get args from a config file and override with cli
@st.cache_data
def get_args(config_file):
    args_from_cli = OmegaConf.from_cli()
    args_from_yaml = OmegaConf.load(config_file)
    return OmegaConf.to_container(OmegaConf.merge(
        args_from_yaml, args_from_cli))


@st.cache_data
def get_splits(args):
    files = os.listdir(os.path.join(
        args['data']['path'], args['data']['dataset']))
    return [split for split in [
        'train', 'val1', 'val2', 'test'] if f'{split}.data' in files]


def get_instance_name(args, row):
    instance = row.to_dict()
    reports = pd.read_csv(io.StringIO(instance['reports']))
    return f'Instance {row.name + 1} (patient {reports.iloc[0].patient_id}, ' \
           f'{len(reports)} reports)'


@st.cache_resource
def get_dataset(args, split):
    df = pd.read_csv(os.path.join(args['data']['path'], args['data']['dataset'], f'{split}.data'), compression='gzip')
    df['instance_name'] = df.apply(lambda r: get_instance_name(args, r), axis=1)
    return df


@st.cache_resource
def get_instance_metadata(_env, args, split):
    df = _env.get_cached_instance_dataframe().reset_index()
    df = df.rename(columns={'index': 'episode_idx'})
    return df


def reset_episode_state():
    del st.session_state['episode']


def get_report_name(row, reference_date=None):
    report_type = row.report_type
    # if row.report_type.strip() == 'Nursing/other':
    report_type += ': \"{}\"'.format(row.text.strip().split('\n')[0])
    date = row.date.strftime('%m/%d/%Y') if reference_date is None else \
        'day {}'.format((row.date - reference_date).days)
    return '{}. {} - {} ({})'.format(
        row.name + 1,
        row.hadm_id,
        report_type,
        date,
    )


def df_from_string(df):
    return pd.read_csv(io.StringIO(df))


def process_reports(reports, reference_row_idx=None):
    kwargs = {}
    if reference_row_idx is not None:
        kwargs['reference_date'] = reports.iloc[reference_row_idx].date
    reports['report_name'] = reports.apply(get_report_name, axis=1, **kwargs)
    return reports


@st.cache_resource
def get_env_models(args):
    return get_model_interface('google/flan-t5-xxl'), \
        SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_environment(args, split, _instances, _llm_interface, _fmm_interface,
                    **override_kwargs):
    kwargs = dict(
        instances=_instances,
        cache_path=args['env'][f'{split}_cache_path'],
        llm_name_or_interface=_llm_interface,
        fmm_name_or_interface=_fmm_interface,
        reward_type=args['env']['reward_type'],
        num_future_diagnoses_threshold=
            args['env']['num_future_diagnoses_threshold'],
        progress_bar=stqdm,
        top_k_evidence=args['env']['top_k_evidence'],
        verbosity=1, # don't print anything when an environment is dead
        add_risk_factor_queries=args['env']['add_risk_factor_queries'],
        limit_options_with_llm=args['env']['limit_options_with_llm'],
        add_none_of_the_above_option=
            args['env']['add_none_of_the_above_option'],
        true_positive_minimum=args['env']['true_positive_minimum'],
        use_confident_diagnosis_mapping=
            args['env']['use_confident_diagnosis_mapping'],
        skip_instances_with_gt_n_reports=
            args['env']['skip_instances_with_gt_n_reports'],
        exclude_evidence=args['env']['exclude_evidence'],
    )
    kwargs.update(override_kwargs)
    return gymnasium.make('ehr_diagnosis_env/EHRDiagnosisEnv-v0', **kwargs)


def get_reset_options(args, i, **kwargs):
    options = {'instance_index': i}
    if args['data']['max_reports_considered'] is not None:
        options['max_reports_considered'] = \
            args['data']['max_reports_considered']
    options.update(kwargs)
    return options


class JumpTo:
    def __init__(self, args, env, jump_to, instance_index):
        self.args = args
        self.env = env
        self.jump_to = jump_to - 1
        self.instance_index = instance_index

    def __call__(self):
        current_timestep = len(st.session_state['episode']['steps'])
        if self.jump_to >= current_timestep:
            st.session_state['episode']['skip_to'] = self.jump_to + 1
            return
        st.session_state['episode']['skip_to'] = None
        for i in range(current_timestep - 1, self.jump_to - 1, -1):
            _, reward, _, _, _ = st.session_state['episode']['steps'][i]
            del st.session_state['episode']['steps'][i]
            del st.session_state['episode']['actions'][i]
        with st.spinner('re-running environment to rewind'):
            # we can do this because we know the environment is not stochastic
            self.env.reset(options=get_reset_options(self.args, self.instance_index))
            for step in range(self.jump_to):
                self.env.step(st.session_state['episode']['actions'][step])


def get_evidence_df(ann_folder):
    rows = {}
    for filename in os.listdir(ann_folder):
        with open(os.path.join(ann_folder, filename), 'rb') as f:
            d = pkl.load(f)
            instance_data = next(iter(d.values()))
            anns = instance_data['model_anns']
            if len(anns) > 0:
                assert len(anns) == 1
                model_type = next(iter(anns.keys()))
                anns['model_type'] = model_type
                model_anns = anns[model_type]
                if len(model_anns) > 0:
                    assert len(model_anns['sort_by_model_order']) == 1
                    sort_type = model_anns['sort_by_model_order'][0]
                    anns['sort_type'] = sort_type
                    del model_anns['sort_by_model_order']
                    model_anns['evidence_anns'] = model_anns['evidence_anns'][sort_type]
                    anns.update(model_anns)
                del anns[model_type]
            del instance_data['model_anns']
            instance_data.update(instance_data['info'])
            del instance_data['info']
            instance_data.update(anns)
            rows.update(d)
    return pd.DataFrame(rows).transpose().sort_index()


def get_full_evidence_df(ann_dirs):
    dfs = defaultdict(lambda : pd.DataFrame([]))
    for ann_dir in ann_dirs:
        date = '_'.join(ann_dir.split('/')[-2].split('_')[1:])
        for split in os.listdir(ann_dir):
            for annotator in os.listdir(os.path.join(ann_dir, split)):
                df = get_evidence_df(os.path.join(ann_dir, split, annotator))
                df['annotator'] = [annotator] * len(df)
                df['date'] = [date] * len(df)
                dfs[split] = pd.concat([dfs[split], df])
    dfs = {split: df.reset_index() for split, df in dfs.items()}
    annotator_instance_repeats = defaultdict(lambda: [])
    for split, df in dfs.items():
        annotations_to_remove = set()
        for annotator in set(df.annotator):
            annotator_df = df[df.annotator == annotator]
            for instance in set(annotator_df.instance):
                annotator_instance_df = annotator_df[
                    annotator_df.instance == instance]
                if len(annotator_instance_df) > 1:
                    annotator_instance_repeats[split].append(annotator_instance_df)
                    annotations_to_remove.update(set(annotator_instance_df[1:].index))
        dfs[split] = df.drop(index=list(annotations_to_remove))
    return dfs, dict(annotator_instance_repeats)


def get_hallucination_annotations(ann_path):
    anns_df = {}
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            with open(os.path.join(ann_path, filename), 'rb') as f:
                anns_df.update(pkl.load(f))
    anns_df = pd.DataFrame(anns_df).transpose()
    return anns_df


def get_all_hallucination_annotations(args, split):
    anns_df = pd.DataFrame([])
    ann_dirs = [args['annotations']['hallucination_anns_path']] \
        + args['annotations']['hallucination_anns_complementary']
    for ann_dir in ann_dirs:
        ann_split_dir = os.path.join(
            ann_dir, split)
        if os.path.exists(ann_split_dir):
            for annotator_path in os.listdir(ann_split_dir):
                anns_df_temp = get_hallucination_annotations(os.path.join(
                    ann_split_dir, annotator_path))
                anns_df_temp['annotator'] = [annotator_path] * len(
                    anns_df_temp)
                anns_df = pd.concat([anns_df, anns_df_temp])
    return anns_df
